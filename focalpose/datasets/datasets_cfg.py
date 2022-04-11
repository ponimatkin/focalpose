from focalpose.config import LOCAL_DATA_DIR, SYNT_DS_DIR
from focalpose.utils.logging import get_logger

from .urdf_dataset import Pix3DUrdfDataset, CarsUrdfDataset
from .texture_dataset import TextureDataset

logger = get_logger(__name__)


def make_scene_dataset(ds_name, n_frames=None):
    is_train = 'train' in ds_name
    # Pix3D
    if 'pix3d' in ds_name and 'synthetic.' not in ds_name:
        from .real_dataset import Pix3DDataset
        if ds_name.lower().split('.')[0] == 'pix3d-sofa':
            ds = Pix3DDataset(ds_dir=LOCAL_DATA_DIR / 'pix3d', category='sofa', train=is_train)
        elif ds_name.lower().split('.')[0] == 'pix3d-chair':
            ds = Pix3DDataset(ds_dir=LOCAL_DATA_DIR / 'pix3d', category='chair', train=is_train)
        elif ds_name.lower().split('.')[0] == 'pix3d-table':
            ds = Pix3DDataset(ds_dir=LOCAL_DATA_DIR / 'pix3d', category='table', train=is_train)
        elif ds_name.lower().split('.')[0] == 'pix3d-bed':
            ds = Pix3DDataset(ds_dir=LOCAL_DATA_DIR / 'pix3d', category='bed', train=is_train)
    elif ds_name.lower().split('.')[0] == 'stanfordcars3d':
        from .real_dataset import StanfordCars3DDataset
        ds = StanfordCars3DDataset(ds_dir=LOCAL_DATA_DIR / 'StanfordCars', train=is_train)
    elif ds_name.lower().split('.')[0] == 'compcars3d':
        from .real_dataset import CompCars3DDataset
        ds = CompCars3DDataset(ds_dir=LOCAL_DATA_DIR / 'CompCars', train=is_train)
    # Synthetic datasets
    elif 'synthetic.' in ds_name:
        assert '.train' in ds_name or '.val' in ds_name
        from .synthetic_dataset import SyntheticSceneDataset
        ds_name = ds_name.split('.')[1]
        ds = SyntheticSceneDataset(ds_dir=SYNT_DS_DIR / ds_name, train=is_train)

    else:
        raise ValueError(ds_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_urdf_dataset(ds_name):
    if ds_name.lower() == 'pix3d-sofa':
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
        mask = (ds.index['category'] == 'sofa')
        ds.index = ds.index[mask].reset_index(drop=True)
    elif ds_name.lower() == 'pix3d-chair':
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
        mask = (ds.index['category'] == 'chair')
        ds.index = ds.index[mask].reset_index(drop=True)
    elif 'pix3d-chair' in ds_name.lower() and 'pix3d-chair-p' not in ds_name.lower():
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
        mask = (ds.index['category'] == 'chair')
        ds.index = ds.index[mask].reset_index(drop=True)

        split_num = int(ds_name.lower().split('-')[-1])
        assert split_num >= 1 and split_num <= 21
        if split_num < 21:
            ds.index = ds.index.iloc[(split_num - 1)*10:split_num*10].reset_index(drop=True)
        else:
            ds.index = ds.index.iloc[(split_num - 1)*10:].reset_index(drop=True)
    elif ds_name.lower() == 'pix3d-table':
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
        mask = (ds.index['category'] == 'table')
        ds.index = ds.index[mask].reset_index(drop=True)
    elif ds_name.lower() == 'pix3d-bed':
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
        mask = (ds.index['category'] == 'bed')
        ds.index = ds.index[mask].reset_index(drop=True)
    elif ds_name.lower() == 'pix3d-sofa-bed-table':
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
        mask = (ds.index['category'] == 'bed') | (ds.index['category'] == 'sofa') | (ds.index['category'] == 'table')
        ds.index = ds.index[mask].reset_index(drop=True)
    elif ds_name.lower() == 'pix3d':
        ds = Pix3DUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'pix3d')
    elif ds_name.lower() == 'stanfordcars3d':
        ds = CarsUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'StanfordCars3D')
    elif 'stanfordcars3d' in ds_name.lower() and 'stanfordcars3d-p' not in ds_name.lower():
        ds = CarsUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'StanfordCars3D')
        split_num = int(ds_name.lower().split('-')[-1])
        assert split_num >= 1 and split_num <= 13
        if split_num < 13:
            ds.index = ds.index.iloc[(split_num - 1)*10:split_num*10].reset_index(drop=True)
        else:
            ds.index = ds.index.iloc[(split_num - 1)*10:].reset_index(drop=True)
    elif ds_name.lower() == 'compcars3d':
        ds = CarsUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'CompCars3D')
    elif 'compcars3d' in ds_name.lower() and 'compcars3d-p' not in ds_name.lower():
        ds = CarsUrdfDataset(root_dir=LOCAL_DATA_DIR / 'models_urdf' / 'CompCars3D')
        split_num = int(ds_name.lower().split('-')[-1])
        assert split_num >= 1 and split_num <= 10
        if split_num < 10:
            ds.index = ds.index.iloc[(split_num - 1)*10:split_num*10].reset_index(drop=True)
        else:
            ds.index = ds.index.iloc[(split_num - 1)*10:].reset_index(drop=True)
    else:
        raise ValueError('Unknown dataset', ds_name)
    return ds


def make_texture_dataset(ds_name):
    if ds_name == 'shapenet':
        ds = TextureDataset(LOCAL_DATA_DIR / 'texture_datasets' / 'shapenet')
    else:
        raise ValueError(ds_name)
    return ds
