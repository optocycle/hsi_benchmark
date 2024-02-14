import os
from collections import namedtuple
from typing import Tuple
import torch
import tqdm as tqdm
from dataloader.s3_config import S3Config
from dataloader.basic_dataloader import HSDataset, load_recording, resize_to_target_size
from camera_definitions import CameraType, get_wavelengths_for
from dataloader.splits import debris_sets
from torch import multiprocessing
from torch.utils.data import get_worker_info
import tempfile

DebrisMetaData = namedtuple('DebrisMetaData', ['class_label', 'camera_type', 'wavelengths', 'filename', 'config'])

# CLASS_LABEL_2_ID_MAPPING = ['background', 'asphalt', 'brick', 'ceramic', 'concrete_light', 'concrete', 'concrete_diverse', 'tile'] # old
CLASS_LABEL_2_ID_MAPPING = ['background', 'asphalt', 'brick', 'concrete', 'tile', 'other']
# CLASS_LABEL_2_ID_MAPPING = ['background', 'concrete', 'natural_stone_grain', 'tile', 'brick', 'sand-lime_brick_and_porous_concrete', 'bituminous_material', 'soil', 'other'] # TODO?!
CLASS_OTHER = ['glass', 'ceramic', 'wood', 'plastic', 'metal', 'paper']

class DebrisDataset(HSDataset):
    def __init__(self, data_path: str, config: str, camera_type: CameraType = CameraType.ALL,
                 split: str = None, balance: bool = False, transform=None,
                 drop_background: bool = True, dilation: int = 1, patch_size: int = None,
                 target_size: Tuple = None,  bucket: str = "home"):
        self.drop_background = drop_background
        self.dilation = dilation
        self.patch_size = patch_size
        self.target_size = target_size
        self.camera_type = camera_type
        self.classes = CLASS_LABEL_2_ID_MAPPING
        self._s3_config = S3Config(endpoint="minio.minio.svc:9000")
        self.bucket = bucket
        self._s3_inst = None
        self._s3_inst_lock = multiprocessing.Lock()
        super().__init__(data_path, config, split, balance, transform)


    def prepare_connection(self):
        s3 = self._s3_config.get_instance()
        self._bucket_region = s3._get_region(self.bucket)
        self._objects = {}
        for obj in s3.list_objects(bucket_name=self.bucket,
                            prefix="j.cicvaric@optocycle.com/Debris/",
                            recursive=True):
            self._objects['/'.join(obj.object_name.split('/')[2:])] = obj.object_name
        return s3

    # TODO move it to a separate class and inherit from it
    def find_hdr_files_in_minio(self, camera_type):
        main_files = []
        for filename in self._objects:
            if filename.endswith("_White.hdr") or filename.endswith("_Dark.hdr"):
                continue
            if filename.endswith('.hdr') and camera_type in filename:
                main_files.append(filename.rstrip('.hdr'))

        return main_files


    def _get_records(self):
        s3 = self.prepare_connection()
        # filter by camera type
        if self.camera_type == CameraType.ALL:
            hdr_files_all = []
            for camera in [CameraType.SPECIM_FX10, CameraType.CORNING_HSI]:
                hdr_files_camera = self.find_hdr_files_in_minio(camera.value)
                hdr_files_all += hdr_files_camera
        else:
            self.data_path = os.path.join(self.data_path, self.camera_type.value)

            hdr_files_all = self.find_hdr_files_in_minio(self.camera_type.value)

        # filter by split
        if self.split is not None:
            assert self.split in ('train', 'val', 'test')
            hdr_files = []
            for hdr_file in hdr_files_all:
                if hdr_file.split('/')[-1] in debris_sets.validation_set:
                    if self.split == 'val':
                        hdr_files.append(hdr_file)
                elif hdr_file.split('/')[-1] in debris_sets.test_set:
                    if self.split == 'test':
                        hdr_files.append(hdr_file)
                elif self.split == 'train':
                    hdr_files.append(hdr_file)
        else:
            hdr_files = hdr_files_all

        records = []
        self._data = {}
        for hdr_file in tqdm.tqdm(hdr_files):
            path_parts = hdr_file.split('/')
            # class_label = path_parts[-1].split('_') # old
            # class_label = '_'.join(class_label[:-2]) # old
            class_label = path_parts[-1].split('_')[0]
            class_label = 'other' if class_label in CLASS_OTHER else class_label
            # TODO check how many s3 connections we actually have
            # TODO we could save this data to disk and load it again without needing to deal with envi files
            if 'CORNING_HSI' in hdr_file:
                camera_type = CameraType.CORNING_HSI
            elif 'SPECIM_FX10' in hdr_file:
                camera_type = CameraType.SPECIM_FX10
            assert camera_type is not None

            if self.patch_size is not None: # patch-wise
                # find pixels
                s3_hdr = s3.get_object(bucket_name=self.bucket, object_name=self._objects[hdr_file + '.hdr'])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.hdr') as temp:
                    temp.write(s3_hdr.data)
                    temp_hdr = temp.name
                s3_bin = s3.get_object(bucket_name=self.bucket, object_name=self._objects[hdr_file + '.bin'])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp:
                    temp.write(s3_bin.data)
                    temp_bin = temp.name
                recording = load_recording(temp_hdr, temp_bin, None)
                recording = torch.tensor(recording)

                pixels = recording.permute(1, 2, 0)
                for x in range(0, len(pixels), self.dilation):
                    row = pixels[x]
                    for y in range(0, len(row), self.dilation):
                        center_pixel = row[y]

                        if center_pixel.max() == center_pixel.min() == 0:
                            if self.drop_background:
                                continue
                            class_label = 'background'

                        records.append(
                            {
                                'filename': hdr_file,
                                'class_label': class_label,
                                'class_id': self.classes.index(class_label),
                                'camera_type': camera_type,
                                'wavelengths': get_wavelengths_for(camera_type),
                            }
                        )
                        x1 = x - (self.patch_size // 2)
                        x2 = x + (self.patch_size // 2) + 1
                        y1 = y - (self.patch_size // 2)
                        y2 = y + (self.patch_size // 2) + 1
                        sample_shape = recording.shape
                        item = torch.zeros((sample_shape[0], self.patch_size, self.patch_size))
                        item[:, -min(0, x1):self.patch_size - max(0, x2 - sample_shape[1]),
                        -min(0, y1):self.patch_size - max(0, y2 - sample_shape[2])] = recording[:, max(x1, 0):min(x2, sample_shape[1]), max(y1, 0):min(sample_shape[2], y2)]
                        item = resize_to_target_size(item, self.target_size)
                        self._data[records[-1]['path']] = item

            else:  # whole image
                records.append(
                    {
                        'filename': hdr_file,
                        'class_label': class_label,
                        'class_id': self.classes.index(class_label),
                        'camera_type': camera_type,
                        'wavelengths': get_wavelengths_for(camera_type)
                    }
                )

        return records

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, object]:
        sample = self.records[index]
        label = torch.tensor(sample['class_id'])
        meta_data = DebrisMetaData(
            class_label=sample['class_label'],
            camera_type=sample['camera_type'],
            wavelengths=sample['wavelengths'],
            filename=sample['filename'],
            config=self.config)

        if self.patch_size is not None:
            
            item = self._data[sample['path']]
        else:

            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            with self._s3_inst_lock:
                if self._s3_inst is None:
                    self._s3_inst = [None for _ in range(worker_info.num_workers if worker_info else 1)]
                if self._s3_inst[worker_id] is None:
                    self._s3_inst[worker_id] = self._s3_config.get_instance(region=self._bucket_region)
            s3 = self._s3_inst[worker_id]

            s3_hdr = s3.get_object(bucket_name=self.bucket, object_name=self._objects[sample['filename'] + '.hdr'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.hdr') as temp:
                temp.write(s3_hdr.data)
                temp_hdr = temp.name
            s3_bin = s3.get_object(bucket_name=self.bucket, object_name=self._objects[sample['filename'] + '.bin'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp:
                temp.write(s3_bin.data)
                temp_bin = temp.name
            item = torch.tensor(load_recording(hdr_name=temp_hdr, bin_name=temp_bin, spatial_size=self.target_size))
            os.remove(temp_bin)
            os.remove(temp_hdr)

        if self.transform is not None:
            item, label, meta_data = self.transform([item, label, meta_data])

        return item, label, meta_data
