import enum
from collections import namedtuple
from typing import Tuple
from scipy import io
import torch
from dataclasses import dataclass
from minio import Minio
import json
from io import BytesIO
from dataloader.basic_dataloader import HSDataset, bands_as_first_dimension, resize_to_target_size
from camera_definitions import CameraType, get_wavelengths_for

class Scene(enum.Enum):
    INDIAN_PINES = 'indian_pines'
    SALINAS = 'salinas'
    PAVIA_UNIVERSITY = 'paviaU'

def str2scene(s: str):
    if s.lower() == 'indian_pines':
        return Scene.INDIAN_PINES
    elif s.lower() == 'salinas':
        return Scene.SALINAS
    elif s.lower() == 'paviau':
        return Scene.PAVIA_UNIVERSITY
    else:
        raise Exception('{} is not a valid remote sensing scene'.format(s))


SCENE_2_LABEL_2_ID_MAPPING = {
    Scene.INDIAN_PINES: ['background', 'alfalfa', 'corn-notill', 'corn-mintill', 'corn', 'grass-pasture', 'grass-trees',
                         'grass-pasture-mowed', 'hay-windrowed', 'oats', 'soybean-notill', 'soybean-montill',
                         'soybean-clean', 'wheat', 'woods', 'buildings-grass-trees-drives', 'stone-steel-towers'],
    Scene.SALINAS: ['background', 'brocoli_green_weeds_1', 'brocoli_green_weeds_2', 'fallow', 'fallow_rough_plow',
                    'fallow_smooth', 'stubble', 'celery', 'grapes_untrained', 'soil_vinyard_develop',
                    'corn_senesced_green_weeds', 'lettuce_romaine_4wk', 'lettuce_romaine_5wk', 'lettuce_romaine_6wk',
                    'lettuce_romaine_7wk', 'vinyard_untrained', 'vinyard_vertical_trellis'],
    Scene.PAVIA_UNIVERSITY: ['background', 'asphalt', 'meadows', 'gravel', 'trees', 'metal_sheets', 'bare soil',
                             'bitumen', 'bricks',
                             'shadows'],
}
SCENE_2_CAMERA_MAPPING = {
    Scene.INDIAN_PINES: CameraType.AVIRIS,
    Scene.SALINAS: CameraType.AVIRIS_2,
    Scene.PAVIA_UNIVERSITY: CameraType.ROSIS
}

RemoteSensingMetaData = namedtuple('RemoteSensingMetaData', ['class_label', 'camera_type', 'wavelengths', 'filename', 'x', 'y', 'config'])


@dataclass
class S3Config:
    endpoint: str
    credentials_file: str = "/home/jovan/.minio/credentials.json"

    @property
    def access_key(self) -> str:
        self._load_credentials()
        return self.__loaded_creds.get("accessKey")

    @property
    def secret_key(self) -> str:
        self._load_credentials()
        return self.__loaded_creds.get("secretKey")

    def _load_credentials(self) -> None:
        if hasattr(self, "__loaded_creds"):
            return
        with open(self.credentials_file) as fp:
            self.__loaded_creds = json.load(fp)

    def get_instance(self, region: str=None) -> Minio:
        s3 = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=region
        )
        return s3


class RemoteSensingDataset(HSDataset):
    def __init__(self, data_path: str, config: str, scene: Scene = Scene.INDIAN_PINES,
                 split: str = None, balance: bool = False, transform=None,
                 drop_invalid: bool = False, dilation: int = 1, patch_size: int = 63, target_size: Tuple = None, train_ratio=0.3):
        self.drop_invalid = drop_invalid
        self.dilation = dilation
        self.patch_size = patch_size
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.scene = scene
        self.classes = SCENE_2_LABEL_2_ID_MAPPING[self.scene]
        self._s3_config = S3Config(endpoint="s3.office.optocycle.net")
        self.bucket = "home"
        self.prepare_connection()
        super().__init__(data_path, config, split, balance, transform)


    def prepare_connection(self):
        self.s3 = self._s3_config.get_instance()
        self._bucket_region = self.s3._get_region(self.bucket)
        self._objects = {}
        for obj in self.s3.list_objects(bucket_name=self.bucket,
                            prefix="j.cicvaric@optocycle.com/hrss_dataset/",
                            recursive=False):
            self._objects[obj.object_name.split('/')[-1]] = obj.object_name


    def _get_records(self):
        
        self.image, self.gt_mask = self.get_image_gt()

        records = []
        self._data = {}

        camera_type = SCENE_2_CAMERA_MAPPING[self.scene]
        wavelengths = get_wavelengths_for(camera_type)

        pixels = self.image.permute(1, 2, 0)
        for x in range(0, len(pixels), self.dilation):
            row = pixels[x]
            for y in range(0, len(row), self.dilation):
                center_label = self.gt_mask[x, y].item()

                if self.drop_invalid and center_label == 0:
                    continue

                records.append(
                    {
                        'path': self.data_path,
                        'filename': self.data_path.split('/')[-1],
                        'class_label': self.classes[center_label],
                        'class_id': center_label,
                        'camera_type': camera_type,
                        'wavelengths': wavelengths,
                        'x1': x - (self.patch_size // 2),
                        'x2': x + (self.patch_size // 2) + 1,
                        'y1': y - (self.patch_size // 2),
                        'y2': y + (self.patch_size // 2) + 1,
                        'xc': x,
                        'yc': y,
                    }
                )

        self._data[records[-1]['path']] = self.image

        if self.split:
            train_records, val_records, test_records = split_train_val_test_set(self.scene, records, train_ratio=self.train_ratio)
            if self.split == 'train':
                return train_records
            elif self.split == 'val':
                return val_records
            elif self.split == 'test':
                return test_records

        return records

    def get_image_gt(self):
        
        if self.scene in [Scene.INDIAN_PINES, Scene.SALINAS]:
            image_name = self.scene.value + '_corrected'
        else:
            image_name = self.scene.value
        image_s3 = self.s3.get_object(bucket_name=self.bucket, object_name=self._objects[image_name[0].upper() + image_name[1:] + '.mat'])
        image = torch.from_numpy(io.loadmat(BytesIO(image_s3.read()))[image_name].astype(float))
        
        image = bands_as_first_dimension(image)
        mask_s3 = self.s3.get_object(bucket_name=self.bucket, object_name=self._objects[self.scene.value[0].upper() + self.scene.value[1:] + '_gt.mat'])
        gt_mask = torch.from_numpy(io.loadmat(BytesIO(mask_s3.read()))[self.scene.value + '_gt'].astype(int))

        return image, gt_mask

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, RemoteSensingMetaData]:
        sample = self.records[index]
        label = sample['class_id']
        meta_data = RemoteSensingMetaData(
            class_label=sample['class_label'],
            camera_type=sample['camera_type'],
            wavelengths=sample['wavelengths'],
            filename=sample['filename'],
            x=sample['xc'],
            y=sample['yc'],
            config=self.config
        )

        x = int(sample['x1'])
        x2 = int(sample['x2'])
        y = int(sample['y1'])
        y2 = int(sample['y2'])
        sample_shape = self._data[sample['path']].shape
        item = torch.zeros((sample_shape[0], self.patch_size, self.patch_size))
        item[:, -min(0, x):self.patch_size - max(0, x2 - sample_shape[1]),
        -min(0, y):self.patch_size - max(0, y2 - sample_shape[2])] = self._data[sample['path']][:,
                                                                     max(x, 0):min(x2, sample_shape[1]),
                                                                     max(y, 0):min(sample_shape[2], y2)]

        if self.target_size is not None:
            item = resize_to_target_size(item, self.target_size)

        if self.transform is not None:
            item, label, meta_data = self.transform([item, label, meta_data])

        return item, label, meta_data


def split_train_val_test_set(scene, records, train_ratio=0.3):
    from dataloader.splits.remote_sensing_pixel_order import INDIAN_PINES_CLASSES, PAVIAU_CLASSES, SALINAS_CLASSES
    total_len = len(records)

    scene_classes = None
    if scene == Scene.INDIAN_PINES:
        scene_classes = INDIAN_PINES_CLASSES
    elif scene == Scene.SALINAS:
        scene_classes = SALINAS_CLASSES
    elif scene == Scene.PAVIA_UNIVERSITY:
        scene_classes = PAVIAU_CLASSES
    else:
        raise RuntimeError(f"Unkown scene {scene}")


    train_pixels, val_pixels, test_pixels = [], [], []
    # train ratio used class-based 
    for scene_class in scene_classes:
        total_len = len(scene_class)

        # 1. split into train and test set by the fixed ratio
        train_len = int(total_len * train_ratio)
        test_len = total_len - train_len

        # 2. split train set into train and val
        val_len = int(train_len * 1/4)
        train_len -= val_len

        train_p = scene_class[:train_len]
        val_p = scene_class[train_len:(train_len + val_len)]
        test_p = scene_class[-test_len:]

        train_pixels += train_p
        val_pixels += val_p
        test_pixels += test_p

    train_pixels = set(train_pixels)
    val_pixels = set(val_pixels)
    test_pixels = set(test_pixels)

    train_records = [r for r in records if ((r['xc'], r['yc']) in train_pixels)]
    val_records = [r for r in records if ((r['xc'], r['yc']) in val_pixels)]
    test_records = [r for r in records if ((r['xc'], r['yc']) in test_pixels)]

    return train_records, val_records, test_records


if __name__ == '__main__':
    import random
    print("# Generate random order of the annotated pixels for a deterministic train-val-test split")
    for scene in (Scene.INDIAN_PINES, Scene.PAVIA_UNIVERSITY, Scene.SALINAS):
        dataset = RemoteSensingDataset("/data/datasets/hrss_dataset", scene=scene)

        print(f"# Scene {scene.value}")
        scene_classes = []
        total_pixels = 0
        for class_id in range(dataset.gt_mask.min(), dataset.gt_mask.max() + 1):
            print(f"# Class {class_id}")
            scene_class_name = scene.value.upper() + "_" + str(class_id)
            l = ((dataset.gt_mask == class_id).nonzero(as_tuple=False).tolist())
            total_pixels += len(l)
            random.shuffle(l)
            print(f"{scene_class_name} = {l}")
            scene_classes.append(scene_class_name)

        print(f"total {total_pixels}")

        print(f"{scene.value.upper()}_CLASSES = [{', '.join(scene_classes)}]")



    
