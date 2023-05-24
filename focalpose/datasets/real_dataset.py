import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

from .utils import make_masks_from_det
from .datasets_cfg import make_urdf_dataset


class Pix3DDataset:
    def __init__(self, ds_dir, category, train=True):
        self.ds_dir = Path(ds_dir)
        self.train = train
        assert self.ds_dir.exists()
        df = pd.read_json((self.ds_dir / 'pix3d.json').as_posix())
        mask = (df['category'] == category) & (df['occluded'] == False) & (df['truncated'] == False) & (
                    df['slightly_occluded'] == False)
        index = df[mask].reset_index(drop=True)

        test_list = self.ds_dir / 'pix3d_test_list.txt'

        test_ids = []
        with test_list.open() as f:
            for i in f:
                test_ids.append('img' + i.replace('\n', ''))

        # Drop images that break our pipeline (1 in each class)
        if category == 'table':
            index = index.drop(index=257)
        elif category == 'sofa':
            index = index.drop(index=76)
        elif category == 'chair':
            index = index.drop(index=2563)
        elif category == 'bed':
            index = index.drop(index=217)
        self.index = index.reset_index(drop=True)

        if self.train:
            mask = ~self.index['img'].isin(test_ids)
        else:
            mask = self.index['img'].isin(test_ids)

        self.index = self.index[mask].reset_index(drop=True)

        if category == 'chair':
            # Fix multiple models in one category
            multiple_models = ['IKEA_JULES_1',
                               'IKEA_MARKUS',
                               'IKEA_PATRIK',
                               'IKEA_SKRUVSTA',
                               'IKEA_SNILLE_1']

            for model in multiple_models:
                mask = self.index['model'].str.contains(model)
                self.index.loc[mask, 'model'] = f'model/chair/{model}/model.obj'

        self.category = category
        self.R_pix3d = R.from_euler('xyz', np.pi * np.array([0, 0, 1]))
        urdf_ds = make_urdf_dataset(f'{ds_dir.as_posix().split("/")[-1]}-{category}')
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        focal_length = entry['focal_length']
        resolution = entry['img_size']
        focal_length = (focal_length * resolution[0]) / 32
        K = np.array([[focal_length, 0, resolution[0] / 2], [0, focal_length, resolution[1] / 2], [0, 0, 1]])
        rgb = np.asarray(Image.open((self.ds_dir / entry['img'])).convert('RGB'))
        mask = np.array(np.asarray(Image.open((self.ds_dir / entry['mask']))) / 255, dtype=np.uint8)

        t = self.R_pix3d.apply(entry['trans_mat']).reshape(3, -1)
        R = self.R_pix3d.as_matrix() @ np.array(entry['rot_mat'])
        TWC = np.linalg.inv(np.vstack([np.hstack([R, t]), [0, 0, 0, 1]]))
        name = entry['model'].replace(f'model/{self.category}/', '').replace('/model.obj',
                                                                             '') + f'_{self.category.upper()}'
        camera = dict(TWC=TWC, K=K, resolution=resolution)
        objects = dict(TWO=np.eye(4), name=name, scale=1, id_in_segm=1, bbox=np.array(entry['bbox']))

        return rgb, mask, dict(camera=camera, objects=[objects])


class StanfordCars3DDataset:
    def __init__(self, ds_dir, train=True):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists()
        self.train = train
        if self.train:
            self.index = pd.read_pickle((self.ds_dir / 'train_anno_preprocessed.pkl').as_posix())
        else:
            self.index = pd.read_pickle((self.ds_dir / 'test_anno_preprocessed.pkl').as_posix())
        urdf_ds = make_urdf_dataset('stanfordcars3d')
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]

    @staticmethod
    def resize_rgb(rgb):
        new_width = 300
        new_height = int(new_width * rgb.size[1] / rgb.size[0])
        return np.asarray(rgb.resize((new_width, new_height), Image.ANTIALIAS))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        K = entry['K']
        TCO = entry['TCO']
        bbox = entry['bbox']
        name = entry['model_id']

        if self.train:
            rgb = self.resize_rgb(Image.open((self.ds_dir / 'cars_train' / entry['img'])).convert('RGB'))
        else:
            rgb = self.resize_rgb(Image.open((self.ds_dir / 'cars_test' / entry['img'])).convert('RGB'))

        mask = make_masks_from_det([np.clip(bbox, a_min=0, a_max=None)], rgb.shape[0], rgb.shape[1]).squeeze(0).numpy()

        camera = dict(TWC=np.linalg.inv(TCO), K=K, resolution=rgb.shape[:2])
        objects = dict(TWO=np.eye(4), name=name, scale=1, id_in_segm=1, bbox=bbox)

        return rgb, mask, dict(camera=camera, objects=[objects])


class CompCars3DDataset:
    def __init__(self, ds_dir, train=True):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists()
        self.train = train
        if self.train:
            self.index = pd.read_pickle((self.ds_dir / 'train_anno_preprocessed.pkl').as_posix())
        else:
            self.index = pd.read_pickle((self.ds_dir / 'test_anno_preprocessed.pkl').as_posix())
        urdf_ds = make_urdf_dataset('compcars3d')
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]

    @staticmethod
    def resize_rgb(rgb):
        new_width = 300
        new_height = int(new_width * rgb.size[1] / rgb.size[0])
        return np.asarray(rgb.resize((new_width, new_height), Image.ANTIALIAS))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        K = entry['K']
        TCO = entry['TCO']
        bbox = entry['bbox']
        name = entry['model_id']

        rgb = self.resize_rgb(Image.open((self.ds_dir / 'data' / 'image' / entry['img'])).convert('RGB'))
        mask = make_masks_from_det([np.clip(bbox, a_min=0, a_max=None)], rgb.shape[0], rgb.shape[1]).squeeze(0).numpy()

        camera = dict(TWC=np.linalg.inv(TCO), K=K, resolution=rgb.shape[:2])
        objects = dict(TWO=np.eye(4), name=name, scale=1, id_in_segm=1, bbox=bbox)

        return rgb, mask, dict(camera=camera, objects=[objects])
