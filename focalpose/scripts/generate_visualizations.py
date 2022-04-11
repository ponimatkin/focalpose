import argparse
import pandas as pd
import numpy as np
import cv2

from focalpose.utils.resources import assign_gpu
from skimage import feature
from skimage import morphology
from skimage import color

from focalpose.config import LOCAL_DATA_DIR
from tqdm import tqdm
from focalpose.rendering.bullet_scene_renderer import BulletSceneRenderer
from focalpose.datasets.datasets_cfg import make_urdf_dataset, make_scene_dataset
from focalpose.datasets.pose_dataset import PoseDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser('StanfordCars3D and CompCars3D preprocessing')
    parser.add_argument('--folder', default='', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--style', default='overlay-contour2d', type=str)
    parser.add_argument('--filter-rot', action='store_true')
    parser.add_argument('--filter-proj', action='store_true')
    args = parser.parse_args()

    assert args.style in ['overlay', 'contour2d', 'contour3d', 'overlay-contour2d']

    assign_gpu()

    if args.filter_rot:
        output_dir = LOCAL_DATA_DIR / 'qualitative_results' / f'{args.folder}_filer_rot'
    elif args.filter_proj:
        output_dir = LOCAL_DATA_DIR / 'qualitative_results' / f'{args.folder}_filer_proj'
    else:
        output_dir = LOCAL_DATA_DIR / 'qualitative_results' / args.folder

    output_dir.mkdir(exist_ok=True, parents=True)

    results = pd.read_pickle((LOCAL_DATA_DIR / 'results' / args.folder / 'results_dataframe.pkl').as_posix())
    scene_ds = make_scene_dataset(args.dataset)

    if 'cars' in args.dataset:
        resize = (200, 300)
    else:
        resize = (640, 640)

    ds_kwargs = dict(
        resize=resize,
        rgb_augmentation=False,
        background_augmentation=False,
        min_area=None,
        gray_augmentation=False,
    )

    ds_eval = PoseDataset(scene_ds, **ds_kwargs)

    urdf_ds = make_urdf_dataset(args.dataset.split('.')[0])
    renderer = BulletSceneRenderer(urdf_ds=args.dataset.split('.')[0], background_color=(255, 255, 255))

    for i in tqdm(range(len(ds_eval)), ncols=80, total=len(ds_eval)):
        if i % 10 == 0:
            renderer.disconnect()
            renderer = BulletSceneRenderer(urdf_ds=args.dataset.split('.')[0], background_color=(255, 255, 255))

        entry = results.iloc[i]
        rgb = ds_eval[i].images

        if args.filter_rot and entry['er'] < np.pi/6:
            continue

        if args.filter_proj and entry['ep'] < 0.1:
            continue

        bbox_pred = entry['bbox_pred']
        bbox_gt = entry['bbox_gt']
        T = entry['T']
        T_coarse = entry['T_coarse']
        T_gt = entry['T_gt']
        T_init = entry['T_init']
        K = entry['K']
        K_init = np.array([[600, 0, 320], [0, 600, 320], [0, 0, 1]])
        K_gt = entry['K_gt']
        name_pred = entry['label_pred']
        name_gt = entry['label_gt']

        obj_infos = [dict(name=name_pred, TWO=np.eye(4))]
        cam_infos = [dict(TWC=np.linalg.inv(T), resolution=resize, K=K)]
        rgb_pred_ren = renderer.render_scene(obj_infos, cam_infos)[0]['rgb']
        mask_pred = renderer.render_scene(obj_infos, cam_infos)[0]['mask']

        if args.style == 'contour3d':
            edges = feature.canny(color.rgb2gray(rgb_pred_ren), sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))
        elif args.style in ['contour2d', 'overlay-contour2d']:
            edges = feature.canny(mask_pred, sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))

        rgb_pred = rgb.copy()
        if args.style in ['contour2d', 'contour3d']:
            rgb_pred[0, edges] = 255
        elif args.style == 'overlay':
            rgb_pred[:, (mask_pred != 255)] = rgb_pred_ren.transpose(2, 0, 1)[:, (mask_pred != 255)]
        elif args.style == 'overlay-contour2d':
            rgb_pred[0, edges] = 255
            rgb_pred[:, (mask_pred != 255)] = rgb_pred_ren.transpose(2, 0, 1)[:, (mask_pred != 255)]

        obj_infos = [dict(name=name_gt, TWO=np.eye(4))]
        cam_infos = [dict(TWC=np.linalg.inv(T_gt), resolution=resize, K=K_gt)]
        rgb_gt_ren = renderer.render_scene(obj_infos, cam_infos)[0]['rgb']
        mask_gt = renderer.render_scene(obj_infos, cam_infos)[0]['mask']

        if args.style == 'contour3d':
            edges = feature.canny(color.rgb2gray(rgb_gt_ren), sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))
        elif args.style in ['contour2d', 'overlay-contour2d']:
            edges = feature.canny(mask_gt, sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))

        rgb_gt = rgb.copy()
        if args.style in ['contour2d', 'contour3d']:
            rgb_gt[1, edges] = 255
        elif args.style == 'overlay':
            rgb_gt[:, (mask_gt != 255)] = rgb_gt_ren.transpose(2, 0, 1)[:, (mask_gt != 255)]
        elif args.style == 'overlay-contour2d':
            rgb_gt[1, edges] = 255
            rgb_gt[:, (mask_gt != 255)] = rgb_gt_ren.transpose(2, 0, 1)[:, (mask_gt != 255)]

        obj_infos = [dict(name=name_pred, TWO=np.eye(4))]
        cam_infos = [dict(TWC=np.linalg.inv(T_init), resolution=resize, K=K_init)]
        rgb_init_ren = renderer.render_scene(obj_infos, cam_infos)[0]['rgb']
        mask_init = renderer.render_scene(obj_infos, cam_infos)[0]['mask']

        if args.style == 'contour3d':
            edges = feature.canny(color.rgb2gray(rgb_init_ren), sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))
        elif args.style in ['contour2d', 'overlay-contour2d']:
            edges = feature.canny(mask_init, sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))

        rgb_init = rgb.copy()
        if args.style in ['contour2d', 'contour3d']:
            rgb_init[2, edges] = 255
        elif args.style == 'overlay':
            rgb_init[:, (mask_init != 255)] = rgb_init_ren.transpose(2, 0, 1)[:, (mask_init != 255)]
        elif args.style == 'overlay-contour2d':
            rgb_init[2, edges] = 255
            rgb_init[:, (mask_init != 255)] = rgb_init_ren.transpose(2, 0, 1)[:, (mask_init != 255)]

        obj_infos = [dict(name=name_pred, TWO=np.eye(4))]
        cam_infos = [dict(TWC=np.linalg.inv(T_coarse), resolution=resize, K=K_init)]
        rgb_coarse_ren = renderer.render_scene(obj_infos, cam_infos)[0]['rgb']
        mask_coarse = renderer.render_scene(obj_infos, cam_infos)[0]['mask']

        if args.style == 'contour3d':
            edges = feature.canny(color.rgb2gray(rgb_coarse_ren), sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))
        elif args.style in ['contour2d', 'overlay-contour2d']:
            edges = feature.canny(mask_coarse, sigma=0)
            edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))

        rgb_coarse = rgb.copy()
        if args.style in ['contour2d', 'contour3d']:
            rgb_coarse[0, edges] = 255
            rgb_coarse[1, edges] = 255
        elif args.style == 'overlay':
            rgb_coarse[:, (mask_coarse != 255)] = rgb_coarse_ren.transpose(2, 0, 1)[:, (mask_coarse != 255)]
        elif args.style == 'overlay-contour2d':
            rgb_coarse[0, edges] = 255
            rgb_coarse[1, edges] = 255
            rgb_coarse[:, (mask_coarse != 255)] = rgb_coarse_ren.transpose(2, 0, 1)[:, (mask_coarse != 255)]

        rgb[0, int(bbox_pred[1]):int(bbox_pred[3]), int(bbox_pred[0]) - 5:int(bbox_pred[0]) + 5] = 255
        rgb[0, int(bbox_pred[1]):int(bbox_pred[3]), int(bbox_pred[2]) - 5:int(bbox_pred[2]) + 5] = 255
        rgb[0, int(bbox_pred[1]) - 5:int(bbox_pred[1]) + 5, int(bbox_pred[0]):int(bbox_pred[2])] = 255
        rgb[0, int(bbox_pred[3]) - 5:int(bbox_pred[3]) + 5, int(bbox_pred[0]):int(bbox_pred[2])] = 255

        rgb[[1, 2], int(bbox_pred[1]):int(bbox_pred[3]), int(bbox_pred[0]) - 5:int(bbox_pred[0]) + 5] = 0
        rgb[[1, 2], int(bbox_pred[1]):int(bbox_pred[3]), int(bbox_pred[2]) - 5:int(bbox_pred[2]) + 5] = 0
        rgb[[1, 2], int(bbox_pred[1]) - 5:int(bbox_pred[1]) + 5, int(bbox_pred[0]):int(bbox_pred[2])] = 0
        rgb[[1, 2], int(bbox_pred[3]) - 5:int(bbox_pred[3]) + 5, int(bbox_pred[0]):int(bbox_pred[2])] = 0

        rgb[1, int(bbox_gt[1]):int(bbox_gt[3]), int(bbox_gt[0]) - 5:int(bbox_gt[0]) + 5] = 255
        rgb[1, int(bbox_gt[1]):int(bbox_gt[3]), int(bbox_gt[2]) - 5:int(bbox_gt[2]) + 5] = 255
        rgb[1, int(bbox_gt[1]) - 5:int(bbox_gt[1]) + 5, int(bbox_gt[0]):int(bbox_gt[2])] = 255
        rgb[1, int(bbox_gt[3]) - 5:int(bbox_gt[3]) + 5, int(bbox_gt[0]):int(bbox_gt[2])] = 255

        rgb[[0, 2], int(bbox_gt[1]):int(bbox_gt[3]), int(bbox_gt[0]) - 5:int(bbox_gt[0]) + 5] = 0
        rgb[[0, 2], int(bbox_gt[1]):int(bbox_gt[3]), int(bbox_gt[2]) - 5:int(bbox_gt[2]) + 5] = 0
        rgb[[0, 2], int(bbox_gt[1]) - 5:int(bbox_gt[1]) + 5, int(bbox_gt[0]):int(bbox_gt[2])] = 0
        rgb[[0, 2], int(bbox_gt[3]) - 5:int(bbox_gt[3]) + 5, int(bbox_gt[0]):int(bbox_gt[2])] = 0

        image = np.concatenate([rgb, rgb_init, rgb_coarse, rgb_pred, rgb_gt], axis=-1).transpose(1, 2, 0)

        cv2.imwrite((output_dir / f'entry_{i + 1}_pred_et_{entry["et"]:.3f}_ef_{entry["ef"]:.3f}_er_{entry["er"]:.3f}.jpeg').as_posix(), image[:, :, [2, 1, 0]])