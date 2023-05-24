import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from focalpose.lib3d.rigid_mesh_database import MeshDataBase
from focalpose.datasets.datasets_cfg import make_urdf_dataset
from focalpose.lib3d.camera_geometry import project_points_robust as project_points
from focalpose.lib3d.camera_geometry import boxes_from_uv
from focalpose.config import LOCAL_DATA_DIR


def get_K(entry, rgb):
    u = entry['u']
    v = entry['v']
    f = entry['f']

    K = np.array([
        [f, 0., rgb.shape[1] // 2 + u],
        [0., f, rgb.shape[0] // 2 + v],
        [0., 0., 1.]])

    return K


if __name__ == "__main__":
    parser = argparse.ArgumentParser('StanfordCars3D and CompCars3D preprocessing')
    parser.add_argument('--dataset', default='', type=str)
    args = parser.parse_args()

    urdf_ds = make_urdf_dataset(args.dataset)
    mesh_db = MeshDataBase.from_urdf_ds(urdf_ds).batched(n_sym=1, resample_n_points=20000).float()
    R_cars = R.from_euler('xyz', np.pi * np.array([0, 0, 1])).as_matrix()

    if args.dataset.lower() == 'stanfordcars3d':
        anno_dict_train = pd.read_pickle(LOCAL_DATA_DIR / 'StanfordCars' / 'train_anno_v2.pkl')
        anno_dict_test = pd.read_pickle(LOCAL_DATA_DIR / 'StanfordCars' / 'test_anno_v2.pkl')
    elif args.dataset.lower() == 'compcars3d':
        anno_dict_train = pd.read_pickle(LOCAL_DATA_DIR / 'CompCars' / 'train_anno.pkl')
        anno_dict_test = pd.read_pickle(LOCAL_DATA_DIR / 'CompCars' / 'test_anno.pkl')

    df_list = list()
    for key in tqdm(anno_dict_train.keys(), ncols=80, total=len(anno_dict_train)):
        entry = anno_dict_train[key]

        if args.dataset.lower() == 'stanfordcars3d':
            rgb = Image.open((LOCAL_DATA_DIR / 'StanfordCars' / 'cars_train' / key)).convert('RGB')
        elif args.dataset.lower() == 'compcars3d':
            rgb = Image.open((LOCAL_DATA_DIR / 'CompCars' / 'data' / 'image' / key)).convert('RGB')

        new_width = 300
        new_height = int(new_width * rgb.size[1] / rgb.size[0])
        rgb = np.asarray(rgb.resize((new_width, new_height), Image.ANTIALIAS))

        t = np.array([0, 0, entry['distance']]).reshape(3, -1)
        Ry = R.from_euler('y', entry['azimuth']).as_matrix()
        Rx = R.from_euler('x', entry['elevation']).as_matrix()
        Rz = R.from_euler('z', entry['theta']).as_matrix()
        TCO = np.vstack([np.hstack([R_cars @ Rz @ Rx.T @ Ry, R_cars @ t]), [0, 0, 0, 1]])
        K = get_K(entry, rgb)

        mesh = mesh_db.select([entry['model_id']]).points
        uv = project_points(mesh, torch.tensor(K).unsqueeze(0).float(), torch.tensor(TCO).unsqueeze(0).float())
        bbox = boxes_from_uv(uv).cpu().numpy()[0]

        entry_dict = dict(
            K=K, TCO=TCO, bbox=bbox, model_id=entry['model_id'], img=key
        )
        df_list.append(entry_dict)

    df = pd.DataFrame(df_list)
    if args.dataset.lower() == 'stanfordcars3d':
        df.to_pickle(LOCAL_DATA_DIR / 'StanfordCars' / 'train_anno_preprocessed.pkl')
    elif args.dataset.lower() == 'compcars3d':
        df.to_pickle(LOCAL_DATA_DIR / 'CompCars' / 'train_anno_preprocessed.pkl')


    df_list = list()
    for key in tqdm(anno_dict_test.keys(), ncols=80, total=len(anno_dict_test)):
        entry = anno_dict_test[key]

        if args.dataset.lower() == 'stanfordcars3d':
            rgb = Image.open((LOCAL_DATA_DIR / 'StanfordCars' / 'cars_test' / key)).convert('RGB')
        elif args.dataset.lower() == 'compcars3d':
            rgb = Image.open((LOCAL_DATA_DIR / 'CompCars' / 'data' / 'image' / key)).convert('RGB')

        new_width = 300
        new_height = int(new_width * rgb.size[1] / rgb.size[0])
        rgb = np.asarray(rgb.resize((new_width, new_height), Image.ANTIALIAS))

        t = np.array([0, 0, entry['distance']]).reshape(3, -1)
        Ry = R.from_euler('y', entry['azimuth']).as_matrix()
        Rx = R.from_euler('x', entry['elevation']).as_matrix()
        Rz = R.from_euler('z', entry['theta']).as_matrix()
        TCO = np.vstack([np.hstack([R_cars @ Rz @ Rx.T @ Ry, R_cars @ t]), [0, 0, 0, 1]])
        K = get_K(entry, rgb)

        mesh = mesh_db.select([entry['model_id']]).points
        uv = project_points(mesh, torch.tensor(K).unsqueeze(0).float(), torch.tensor(TCO).unsqueeze(0).float())
        bbox = boxes_from_uv(uv).cpu().numpy()[0]

        entry_dict = dict(
            K=K, TCO=TCO, bbox=bbox, model_id=entry['model_id'], img=key
        )
        df_list.append(entry_dict)

    df = pd.DataFrame(df_list)
    if args.dataset.lower() == 'stanfordcars3d':
        df.to_pickle(LOCAL_DATA_DIR / 'StanfordCars' / 'test_anno_preprocessed.pkl')
    elif args.dataset.lower() == 'compcars3d':
        df.to_pickle(LOCAL_DATA_DIR / 'CompCars' / 'test_anno_preprocessed.pkl')