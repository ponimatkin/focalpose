import yaml
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision import transforms as pth_transforms

from scipy.linalg import logm

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from focalpose.config import EXP_DIR, LOCAL_DATA_DIR, FEATURES_DIR
from focalpose.utils.resources import assign_gpu
from focalpose.utils.logging import get_logger
from focalpose.rendering.bullet_batch_renderer import BulletBatchRenderer
from focalpose.datasets.datasets_cfg import make_urdf_dataset, make_scene_dataset
from focalpose.datasets.pose_dataset import PoseDataset
from focalpose.models.mask_rcnn import DetectorMaskRCNN
from focalpose.lib3d.transform_ops import transform_pts
from focalpose.lib3d.focalpose_ops import TCO_init_from_boxes_zup_autodepth
from focalpose.lib3d.camera_geometry import project_points_robust as project_points
from focalpose.training.pose_models_cfg import create_model_pose
from focalpose.lib3d.rigid_mesh_database import MeshDataBase
from focalpose.models import vision_transformer as vits


cudnn.benchmark = False

logger = get_logger(__name__)


def overlay_figure(plotter, rgb, scene_rgb):
    scene_rgb = np.asarray(scene_rgb)
    rgb = np.asarray(rgb)
    final_rgba = np.asarray(rgb).copy()
    final_rgba *= 0
    mask = ~(scene_rgb.sum(axis=-1) == (255 * 3))
    final_rgba[~mask] = rgb[~mask] * 0.4 + 255 * 0.6
    final_rgba[mask] = scene_rgb[mask] * 0.9 + 255 * 0.1
    f = plotter.plot_image(final_rgba, name='image')
    return f


def cast(obj):
    return obj.cuda(non_blocking=True)


def load_pose_model(run_id, cfg, batch_renderer, mesh_db):
    model = create_model_pose(cfg=cfg, renderer=batch_renderer, mesh_db=mesh_db).cuda().float()
    pth_dir = EXP_DIR / run_id
    path = pth_dir / 'checkpoint.pth.tar'
    logger.info(f'Loading model from {path}')
    save = torch.load(path)
    state_dict = save['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_detector_model(run_id, cfg, label_to_category_id):
    model = DetectorMaskRCNN(input_resize=cfg.input_resize,
                             n_classes=len(label_to_category_id),
                             backbone_str=cfg.backbone_detector,
                             anchor_sizes=cfg.anchor_sizes).cuda().float()
    pth_dir = EXP_DIR / run_id
    path = pth_dir / 'checkpoint.pth.tar'
    logger.info(f'Loading model from {path}')
    model.load_state_dict(torch.load(path.as_posix())['state_dict'])
    model.eval()
    return model

def evaluate(cfg):
    coarse_run_id = cfg.coarse_run_id
    output_dir = LOCAL_DATA_DIR / 'results' / f'{coarse_run_id}_mrcnn_detections_niter={cfg.niter}'

    output_dir.mkdir(exist_ok=True, parents=True)
    urdf_ds = make_urdf_dataset(cfg.dataset.split('.')[0])
    mesh_db = MeshDataBase.from_urdf_ds(urdf_ds).batched().cuda().float()
    batch_renderer = BulletBatchRenderer(object_set=cfg.dataset.split('.')[0], n_workers=cfg.n_rendering_workers, preload_cache=False, split_objects=True)

    ds_kwargs = dict(
        resize=cfg.input_resize,
        rgb_augmentation=False,
        background_augmentation=False,
        min_area=None,
        gray_augmentation=False,
    )

    train_transform = pth_transforms.Compose([
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transform = pth_transforms.Compose([
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    scene_ds_train = make_scene_dataset(cfg.dataset.replace('test', 'train'))
    scene_ds_eval = make_scene_dataset(cfg.dataset)

    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_eval = PoseDataset(scene_ds_eval, **ds_kwargs)

    ds_iter_eval = DataLoader(ds_eval, batch_size=32, num_workers=cfg.n_dataloader_workers,
                              collate_fn=ds_eval.collate_fn, drop_last=False, pin_memory=False, shuffle=False)

    print(f"Generating bounding boxes for the dataset: {args.dataset}")
    label_to_category_id = dict()
    label_to_category_id['background'] = 0
    label_to_category_id['object'] = 1

    coarse_model = load_pose_model(cfg.coarse_run_id, cfg, batch_renderer, mesh_db)
    refine_model = load_pose_model(cfg.refine_run_id, cfg, batch_renderer, mesh_db)
    mrcnn_model = load_detector_model(cfg.mrcnn_run_id, cfg, label_to_category_id)

    pred_bboxes = []
    for data in tqdm(ds_iter_eval, ncols=80):
        batch_size, _, h, w = data.images.shape
        images = list(cast(image).float() / 255 for image in data.images)
        with torch.no_grad():
            det_outputs = mrcnn_model(images)

        for det_output in det_outputs:
            try:
                pred_bboxes.append(det_output['boxes'][0])
            except IndexError:
                if 'cars' in cfg.dataset:
                    pred_bboxes.append(cast(torch.tensor([0, 0, 200, 300])).float())
                else:
                    pred_bboxes.append(cast(torch.tensor([0, 0, 640, 640])).float())

    pred_bboxes = torch.stack(pred_bboxes)

    del mrcnn_model

    vit_model = vits.__dict__['vit_base'](patch_size=8, num_classes=0)
    for p in vit_model.parameters():
        p.requires_grad = False
    vit_model.eval()
    vit_model.to('cuda')

    vit_model.load_state_dict(torch.load(LOCAL_DATA_DIR / 'dino_vitbase8_pretrain.pth'), strict=True)

    labels = set(scene_ds_train.all_labels)
    labels = labels.union(scene_ds_eval.all_labels)
    labels = sorted(list(labels))
    labels_to_id = {}
    id_to_labels = {}
    for idx, label in enumerate(labels):
        labels_to_id[label] = idx
        id_to_labels[idx] = label

    if not (FEATURES_DIR / (args.dataset.split('.')[0] + '_train_features.npy')).is_file():
        print(f"Extracting DINO features for the dataset: {args.dataset}")
        features_train = []
        y_train = []
        for entry in tqdm(ds_train, total=len(ds_train)):
            img = torch.tensor(entry.images).float().cuda()

            dilated_bbox = torch.tensor([entry.bboxes[0] - entry.bboxes[0] * 0.2, entry.bboxes[1] - entry.bboxes[1] * 0.2,
                                         entry.bboxes[2] + entry.bboxes[2] * 0.2, entry.bboxes[3] + entry.bboxes[3] * 0.2])
            dilated_bbox = dilated_bbox.unsqueeze(0).float().cuda()

            img = torchvision.ops.roi_align(img.unsqueeze(0), [dilated_bbox], output_size=(224, 224))[0] / 255
            img = train_transform(img.cpu()).cuda()
            fts = vit_model(img.unsqueeze(0)).cpu().numpy()
            y_train.append(labels_to_id[entry.objects['name']])
            features_train.append(fts[0])

        X_train = np.array(features_train)
        y_train = np.array(y_train)
        np.save(FEATURES_DIR / (args.dataset.split('.')[0] + '_train_features.npy'), X_train)
        np.save(FEATURES_DIR / (args.dataset.split('.')[0] + '_train_y.npy'), y_train)
    else:
        print(f"Using cached DINO features for the dataset: {args.dataset}")
        X_train = np.load(FEATURES_DIR / (args.dataset.split('.')[0] + '_train_features.npy'))
        y_train = np.load(FEATURES_DIR / (args.dataset.split('.')[0] + '_train_y.npy'))

    if not (FEATURES_DIR / (args.dataset.split('.')[0] + '_test_features.npy')).is_file():
        features_test = []
        y_test = []
        for entry, bbox in tqdm(zip(ds_eval, pred_bboxes), total=len(ds_eval)):
            img = torch.tensor(entry.images).float().cuda()

            img = torchvision.ops.roi_align(img.unsqueeze(0), [bbox.unsqueeze(0)], output_size=(224, 224))[0] / 255
            img = val_transform(img.cpu()).cuda()

            fts = vit_model(img.unsqueeze(0)).cpu().numpy()
            y_test.append(labels_to_id[entry.objects['name']])

            features_test.append(fts[0])

        X_test = np.array(features_test)
        y_test = np.array(y_test)
        np.save(FEATURES_DIR / (args.dataset.split('.')[0] + '_test_features.npy'), X_test)
        np.save(FEATURES_DIR / (args.dataset.split('.')[0] + '_test_y.npy'), y_test)
    else:
        X_test = np.load(FEATURES_DIR / (args.dataset.split('.')[0] + '_test_features.npy'))
        y_test = np.load(FEATURES_DIR / (args.dataset.split('.')[0] + '_test_y.npy'))

    print('Learning the classfier over the DINO features')

    le = LabelEncoder()
    le.fit(y_train)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    clf = make_pipeline(StandardScaler(), VotingClassifier(estimators=[
        ('clf1', LogisticRegression(tol=1e-10, n_jobs=-1, penalty='l2', C=10, solver='lbfgs', max_iter=100000)),
        ('clf2', LogisticRegression(tol=1e-10, n_jobs=-1, penalty='l2', C=10, solver='lbfgs', max_iter=100000)),
        ('clf3', LogisticRegression(tol=1e-10, n_jobs=-1, penalty='l2', C=10, solver='lbfgs', max_iter=100000)),
        ('clf4', LogisticRegression(tol=1e-10, n_jobs=-1, penalty='l2', C=10, solver='lbfgs', max_iter=100000)),
        ('clf5', LogisticRegression(tol=1e-10, n_jobs=-1, penalty='l2', C=10, solver='lbfgs', max_iter=100000))], voting='soft'))
    clf.fit(X_train, y_train)

    top_5 = []
    y_pred = clf.predict_proba(X_test)

    for idx in range(len(y_pred)):
        if y_test[idx] in np.argsort(y_pred[idx])[::-1][:5]:
            top_5.append(1)
        else:
            top_5.append(0)

    print(f'Top-1 performance: {clf.score(X_test, y_test)}')
    print(f'Top-5 Performance: {np.mean(top_5)}\n')

    del vit_model

    pred_labels_all = np.array([id_to_labels[le.inverse_transform([idx]).item()] for idx in clf.predict(X_test)])

    acc = []
    er = []
    et = []
    ert = []
    ep = []
    ef = []
    eret = []

    result_dicts = list()

    for idx, data in tqdm(enumerate(ds_iter_eval), ncols=80, total=len(ds_iter_eval)):
        batch_size, _, h, w = data.images.shape
        images = cast(data.images).float() / 255.

        K_gt = cast(data.K).float()
        TCO_gt = cast(data.TCO).float()
        bboxes_gt = cast(data.bboxes).float()
        labels_gt = np.array([obj['name'] for obj in data.objects])


        pred_labels = pred_labels_all[idx*32:(idx+1)*32]
        pred_bbox = pred_bboxes[idx*32:(idx+1)*32]

        points = mesh_db.select(labels_gt).points
        points_init = mesh_db.select(pred_labels).points

        K_init = K_gt.clone()
        K_init[:, 0, 0] = 600
        K_init[:, 1, 1] = 600

        TCO_init = TCO_init_from_boxes_zup_autodepth(model_points_3d=points_init, boxes_2d=pred_bbox, K=K_init)

        if cfg.niter > 0:
            with torch.no_grad():
                outputs = coarse_model(images=images, K=K_init, labels=pred_labels,
                                           TCO=TCO_init, n_iterations=1, update_focal_length=True)
            iter_outputs = outputs[f'iteration={1}']
            TCO_coarse = iter_outputs['TCO_output']
            K_init = iter_outputs['K_output']

            with torch.no_grad():
                outputs = refine_model(images=images, K=K_init, labels=pred_labels,
                                       TCO=TCO_coarse, n_iterations=cfg.niter, update_focal_length=True)

            iter_outputs = outputs[f'iteration={cfg.niter}']
            TCO_pred = iter_outputs['TCO_output']
            K_pred = iter_outputs['K_output']
        else:
            with torch.no_grad():
                outputs = coarse_model(images=images, K=K_init, labels=pred_labels,
                                           TCO=TCO_init, n_iterations=1, update_focal_length=True)

            iter_outputs = outputs[f'iteration={1}']
            TCO_pred = iter_outputs['TCO_output']
            TCO_coarse = iter_outputs['TCO_output']
            K_pred = iter_outputs['K_output']

        pts_pred_tr = transform_pts(TCO_pred, points)
        pts_gt_tr = transform_pts(TCO_gt, points)
        pts_pred_proj = project_points(points, K_pred, TCO_pred)
        pts_gt_proj = project_points(points, K_gt, TCO_gt)

        ef.extend(np.abs(K_gt[:, 0, 0].cpu().numpy() - K_pred[:, 0, 0].cpu().numpy()) / np.abs(K_gt[:, 0, 0].cpu().numpy()))

        for idx in range(batch_size):
            if labels_gt[idx] == pred_labels[idx]:
                eret.append(1)
            else:
                eret.append(0)

            R_pred = TCO_pred[idx, :3, :3].cpu()
            t_pred = TCO_pred[idx, :3, 3].cpu().numpy()
            R_gt = TCO_gt[idx, :3, :3].cpu()
            t_gt = TCO_gt[idx, :3, 3].cpu().numpy()
            bbox_gt = bboxes_gt[idx].cpu().numpy()

            acc.append(box_iou(pred_bbox[idx].cpu().unsqueeze(0).float(), bboxes_gt[idx].cpu().unsqueeze(0).float())[0].item())

            d_img = np.sqrt(w**2 + h**2)
            d_bbox = np.sqrt((bbox_gt[2] - bbox_gt[0])**2 + (bbox_gt[3] - bbox_gt[1])**2)

            er.append(np.linalg.norm(logm(R_gt.numpy().T @ R_pred.numpy(), disp=False)[0], ord='fro') / np.sqrt(2))
            et.append(np.linalg.norm(t_gt - t_pred)/np.linalg.norm(t_gt))
            ert.append((d_bbox/d_img)*(torch.norm(pts_pred_tr[idx] - pts_gt_tr[idx], dim=-1).mean().cpu().numpy()/np.linalg.norm(t_gt)))
            ep.append(torch.norm(pts_pred_proj[idx] - pts_gt_proj[idx], dim=-1).mean().cpu().numpy()/d_bbox)

            result_dicts.append(
                dict(
                    T=TCO_pred[idx].cpu().numpy(),
                    K=K_pred[idx].cpu().numpy(),
                    T_gt=TCO_gt[idx].cpu().numpy(),
                    T_init=TCO_init[idx].cpu().numpy(),
                    T_coarse=TCO_coarse[idx].cpu().numpy(),
                    K_gt=K_gt[idx].cpu().numpy(),
                    bbox_pred=pred_bbox[idx].cpu().numpy(),
                    bbox_gt=bboxes_gt[idx].cpu().numpy(),
                    ef=np.abs(K_gt[idx, 0, 0].cpu().numpy() - K_pred[idx, 0, 0].cpu().numpy()) / np.abs(K_gt[idx, 0, 0].cpu().numpy()),
                    er=np.linalg.norm(logm(R_gt.numpy().T @ R_pred.numpy(), disp=False)[0], ord='fro') / np.sqrt(2),
                    et=np.linalg.norm(t_gt - t_pred)/np.linalg.norm(t_gt),
                    ert=(d_bbox/d_img)*(torch.norm(pts_pred_tr[idx] - pts_gt_tr[idx], dim=-1).mean().cpu().numpy()/np.linalg.norm(t_gt)),
                    ep=torch.norm(pts_pred_proj[idx] - pts_gt_proj[idx], dim=-1).mean().cpu().numpy()/d_bbox,
                    label_gt=labels_gt[idx],
                    label_pred=pred_labels[idx]
                )
            )


    log_dict = dict()
    log_dict['R_err_median'] = np.median(er).item()
    log_dict['t_err_median'] = np.median(et).item()
    log_dict['f_err_median'] = np.median(ef).item()
    log_dict['R_acc_30_deg'] = np.mean(np.array(er) <= np.radians(30)).item()
    log_dict['R_acc_15_deg'] = np.mean(np.array(er) <= np.radians(15)).item()
    log_dict['R_acc_5_deg'] = np.mean(np.array(er) <= np.radians(5)).item()
    log_dict['proj_acc_0.1'] = np.mean(np.array(ep) <= 0.1).item()
    log_dict['proj_acc_0.05'] = np.mean(np.array(ep) <= 0.05).item()
    log_dict['proj_acc_0.01'] = np.mean(np.array(ep) <= 0.01).item()
    log_dict['Rt_err_median'] = np.median(ert).item()
    log_dict['proj_err_median'] = np.median(ep).item()
    log_dict['acc_0.5'] = np.mean(np.array(acc) > 0.5).item()
    log_dict['ret_acc'] = np.mean(eret).item()
    logger.info(log_dict)

    output_file = output_dir / 'results.yaml'
    with output_file.open('w') as fp:
        yaml.safe_dump(log_dict, fp, indent=4, default_flow_style=False)

    np.savetxt((output_dir / 'er_hist.txt').as_posix(), er)
    np.savetxt((output_dir / 'et_hist.txt').as_posix(), et)
    np.savetxt((output_dir / 'ef_hist.txt').as_posix(), ef)
    np.savetxt((output_dir / 'ep_hist.txt').as_posix(), ep)
    np.savetxt((output_dir / 'ert_hist.txt').as_posix(), ert)

    df = pd.DataFrame(result_dicts)
    df.to_pickle((output_dir / 'results_dataframe.pkl').as_posix())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pose evaluation')
    parser.add_argument('--coarse-run-id', default='', type=str)
    parser.add_argument('--refine-run-id', default='', type=str)
    parser.add_argument('--mrcnn-run-id', default='', type=str)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--dataset', default='', type=str)
    args = parser.parse_args()
    cfg = argparse.ArgumentParser('').parse_args([])

    assign_gpu()

    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    cfg.dataset = args.dataset
    if 'cars' in cfg.dataset:
        cfg.input_resize = (200, 300)
    else:
        cfg.input_resize = (640, 640)

    # Detection Model
    cfg.mrcnn_run_id = args.mrcnn_run_id
    cfg.backbone_detector = 'resnet50-fpn'
    cfg.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    # Pose Model
    cfg.backbone_str = 'resnet50'
    cfg.backbone_pretrained = True
    cfg.coarse_run_id = args.coarse_run_id
    cfg.refine_run_id = args.refine_run_id
    cfg.mrcnn_run_id = args.mrcnn_run_id
    cfg.n_pose_dims = 9
    if 'cars' in cfg.dataset:
        cfg.n_rendering_workers = 16
    elif 'chair' in cfg.dataset:
        cfg.n_rendering_workers = 16
    else:
        cfg.n_rendering_workers = 8
    cfg.n_dataloader_workers = 8
    cfg.niter = args.niter

    evaluate(cfg=cfg)