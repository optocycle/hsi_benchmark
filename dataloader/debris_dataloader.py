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
        self._s3_config = S3Config(endpoint="s3.office.optocycle.net")
        self.bucket = bucket
        self._s3_inst = None
        self._s3_inst_lock = multiprocessing.Lock()

        self.prepare_connection()
        super().__init__(data_path, config, split, balance, transform)


    # TODO move it to a separate class and inherit from it
    def prepare_connection(self):
        s3 = self._s3_config.get_instance()
        self._bucket_region = s3._get_region(self.bucket)
        self._objects = {}
        for obj in s3.list_objects(bucket_name=self.bucket,
                            prefix="j.cicvaric@optocycle.com/Debris/",
                            recursive=True):
            self._objects['/'.join(obj.object_name.split('/')[2:])] = obj.object_name


    # TODO move it to a separate class and inherit from it
    def find_hdr_files_in_minio(self, camera_type):
        main_files = []
        for filename in self._objects:
            if filename.endswith("_White.hdr") or filename.endswith("_Dark.hdr"):
                continue
            if filename.endswith('.hdr') and camera_type in filename:
                main_files.append(filename)

        return main_files


    def _get_records(self):
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
                if hdr_file.split('/')[-1].split('.')[0] in debris_sets.validation_set:
                    if self.split == 'val':
                        hdr_files.append(hdr_file)
                elif hdr_file.split('/')[-1].split('.')[0] in debris_sets.test_set:
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
            # TODO rewrite

            if 'CORNING_HSI' in os.path.join(self.data_path, hdr_file):
                camera_type = CameraType.CORNING_HSI
            elif 'SPECIM_FX10' in os.path.join(self.data_path, hdr_file):
                camera_type = CameraType.SPECIM_FX10
            assert camera_type is not None
            path = os.path.join(self.data_path, hdr_file)

            if self.patch_size is not None: # patch-wise
                # find pixels
                recording = load_recording(path, None)
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
                                'path': path,
                                'filename': hdr_file,
                                'class_label': class_label,
                                'class_id': self.classes.index(class_label),
                                'camera_type': camera_type,
                                'wavelengths': get_wavelengths_for(camera_type),
                                'x1': x - (self.patch_size // 2),
                                'x2': x + (self.patch_size // 2) + 1,
                                'y1': y - (self.patch_size // 2),
                                'y2': y + (self.patch_size // 2) + 1,
                            }
                        )

                        self._data[records[-1]['path']] = recording

            else:  # whole image
                records.append(
                    {
                        'path': path,
                        'filename': hdr_file,
                        'class_label': class_label,
                        'class_id': self.classes.index(class_label),
                        'camera_type': camera_type,
                        'wavelengths': get_wavelengths_for(camera_type)
                    }
                )

        return records

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, object]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        with self._s3_inst_lock:
            if self._s3_inst is None:
                self._s3_inst = [None for _ in range(worker_info.num_workers if worker_info else 1)]
            if self._s3_inst[worker_id] is None:
                self._s3_inst[worker_id] = self._s3_config.get_instance(region=self._bucket_region)
        s3 = self._s3_inst[worker_id]

        
        sample = self.records[index]
        label = torch.tensor(sample['class_id'])
        meta_data = DebrisMetaData(
            class_label=sample['class_label'],
            camera_type=sample['camera_type'],
            wavelengths=sample['wavelengths'],
            filename=sample['filename'],
            config=self.config)

        if self.patch_size is not None:
            x = int(sample['x1'])
            x2 = int(sample['x2'])
            y = int(sample['y1'])
            y2 = int(sample['y2'])
            sample_shape = self._data[sample['path']].shape
            item = torch.zeros((sample_shape[0], self.patch_size, self.patch_size))
            item[:, -min(0, x):self.patch_size - max(0, x2 - sample_shape[1]),
            -min(0, y):self.patch_size - max(0, y2 - sample_shape[2])] = self._data[sample['path']][:, max(x, 0):min(x2, sample_shape[1]), max(y, 0):min(sample_shape[2], y2)]
            item = resize_to_target_size(item, self.target_size)
        else:
            # bad and slow solution
            s3.fget_object(bucket_name=self.bucket, object_name=self._objects[sample['filename']], file_path=sample['filename'].replace('/', '_'))
            s3.fget_object(bucket_name=self.bucket, object_name=self._objects[sample['filename'].replace('.hdr', '.bin')], file_path=sample['filename'].replace('/', '_').replace('.hdr', '.bin'))
            item = torch.tensor(load_recording(sample['filename'].replace('/', '_').rstrip('.hdr'), spatial_size=self.target_size))
            os.remove(sample['filename'].replace('/', '_'))
            os.remove(sample['filename'].replace('/', '_').replace('.hdr', '.bin'))

        if self.transform is not None:
            item, label, meta_data = self.transform([item, label, meta_data])

        return item, label, meta_data
