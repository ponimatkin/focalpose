import cv2
import torch
import argparse
import torchvision
import numpy as np

from tqdm import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms

from skimage import feature
from skimage import morphology
from skimage import color

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from focalpose.config import EXP_DIR, LOCAL_DATA_DIR, FEATURES_DIR
from focalpose.utils.resources import assign_gpu
from focalpose.utils.logging import get_logger
from focalpose.rendering.bullet_batch_renderer import BulletBatchRenderer
from focalpose.rendering.bullet_scene_renderer import BulletSceneRenderer
from focalpose.datasets.datasets_cfg import make_urdf_dataset, make_scene_dataset
from focalpose.datasets.pose_dataset import PoseDataset
from focalpose.models.mask_rcnn import DetectorMaskRCNN
from focalpose.lib3d.focalpose_ops import TCO_init_from_boxes_zup_autodepth
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

    if cfg.cls in ['bed', 'sofa', 'table', 'chair']:
        ds_name = f'pix3d-{cfg.cls}.test.gt'
        coarse_model = f'pix3d-{cfg.cls}-coarse-F05p-disent--cvpr2022'
        refine_model = f'pix3d-{cfg.cls}-refine-F05p-disent--cvpr2022'
        detector_model = f'detector-pix3d-{cfg.cls}-real-two-class--cvpr2022'
    elif cfg.cls == 'compcars':
        ds_name = f'compcars3d.test.gt'
        coarse_model = f'compcars3d-coarse-F05p-disent--cvpr2022'
        refine_model = f'compcars3d-refine-F05p-disent--cvpr2022'
        detector_model = f'detector-compcars3d-real-two-class--cvpr2022'
    elif cfg.cls == 'stanfordcars':
        ds_name = f'stanfordcars3d.test.gt'
        coarse_model = f'stanfordcars3d-coarse-F05p-disent--cvpr2022'
        refine_model = f'stanfordcars3d-refine-F05p-disent--cvpr2022'
        detector_model = f'detector-stanfordcars3d-real-two-class--cvpr2022'

    urdf_ds = make_urdf_dataset(ds_name.split('.')[0])
    mesh_db = MeshDataBase.from_urdf_ds(urdf_ds).batched().cuda().float()
    batch_renderer = BulletBatchRenderer(object_set=ds_name.split('.')[0], n_workers=cfg.n_rendering_workers, preload_cache=False, split_objects=True)

    target_im = cv2.imread(cfg.img)
    target_im = cv2.resize(target_im, (cfg.input_resize[1], cfg.input_resize[0]))[..., ::-1]

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

    scene_ds_train = make_scene_dataset(ds_name.replace('test', 'train'))
    scene_ds_eval = make_scene_dataset(ds_name)

    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_eval = PoseDataset(scene_ds_eval, **ds_kwargs)

    ds_iter_eval = DataLoader(ds_eval, batch_size=32, num_workers=cfg.n_dataloader_workers,
                              collate_fn=ds_eval.collate_fn, drop_last=False, pin_memory=False, shuffle=False)

    label_to_category_id = dict()
    label_to_category_id['background'] = 0
    label_to_category_id['object'] = 1

    coarse_model = load_pose_model(coarse_model, cfg, batch_renderer, mesh_db)
    refine_model = load_pose_model(refine_model, cfg, batch_renderer, mesh_db)
    mrcnn_model = load_detector_model(detector_model, cfg, label_to_category_id)

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

    if not (FEATURES_DIR / (ds_name.split('.')[0] + '_train_features.npy')).is_file():
        print(f"Generating bounding boxes for the dataset: {ds_name}")
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
                    if 'cars' in cfg.cls:
                        pred_bboxes.append(cast(torch.tensor([0, 0, 200, 300])).float())
                    else:
                        pred_bboxes.append(cast(torch.tensor([0, 0, 640, 640])).float())

        pred_bboxes = torch.stack(pred_bboxes)

        print(f"Extracting DINO features for the dataset: {ds_name}")
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
        np.save(FEATURES_DIR / (ds_name.split('.')[0] + '_train_features.npy'), X_train)
        np.save(FEATURES_DIR / (ds_name.split('.')[0] + '_train_y.npy'), y_train)
    else:
        print(f"Using cached DINO features for the dataset: {ds_name}")
        X_train = np.load(FEATURES_DIR / (ds_name.split('.')[0] + '_train_features.npy'))
        y_train = np.load(FEATURES_DIR / (ds_name.split('.')[0] + '_train_y.npy'))

    if not (FEATURES_DIR / (ds_name.split('.')[0] + '_test_features.npy')).is_file():
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
        np.save(FEATURES_DIR / (ds_name.split('.')[0] + '_test_features.npy'), X_test)
        np.save(FEATURES_DIR / (ds_name.split('.')[0] + '_test_y.npy'), y_test)
    else:
        X_test = np.load(FEATURES_DIR / (ds_name.split('.')[0] + '_test_features.npy'))
        y_test = np.load(FEATURES_DIR / (ds_name.split('.')[0] + '_test_y.npy'))

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

    pred_bbox = []
    images = [cast(torch.tensor(target_im.copy())).float().permute(2, 0, 1) / 255]
    with torch.no_grad():
        det_outputs = mrcnn_model(images)

    for det_output in det_outputs:
        try:
            pred_bbox.append(det_output['boxes'][0])
        except IndexError:
            if 'cars' in cfg.dataset:
                pred_bbox.append(cast(torch.tensor([0, 0, 200, 300])).float())
            else:
                pred_bbox.append(cast(torch.tensor([0, 0, 640, 640])).float())

    pred_bbox = torch.stack(pred_bbox)
    img = torch.tensor(target_im.copy()).float().cuda().permute(2, 0, 1)
    img = torchvision.ops.roi_align(img.unsqueeze(0), [pred_bbox[0].unsqueeze(0)], output_size=(224, 224))[0] / 255
    img = val_transform(img.cpu()).cuda()
    fts = vit_model(img.unsqueeze(0)).cpu().numpy()

    model_cls_probs = clf.predict_proba(fts.reshape(1, -1))
    model_cls_ids = np.argsort(model_cls_probs, axis=1)[:, -cfg.topk:]
    model_labels = [id_to_labels[le.inverse_transform([model_id]).item()] for model_id in model_cls_ids[0]]

    del vit_model
    del mrcnn_model

    renderer = BulletSceneRenderer(urdf_ds=ds_name.split('.')[0], background_color=(255, 255, 255))
    for model_id, model_label in enumerate(model_labels):
        images = cast(torch.tensor(target_im.copy())).float().permute(2, 0, 1)[None] / 255.
        points_init = mesh_db.select([model_label]).points

        K_init = np.array([[600, 0, cfg.input_resize[0] / 2], [0, 600, cfg.input_resize[1] / 2], [0, 0, 1]])
        K_init = torch.tensor(K_init)[None].float().cuda()

        TCO_init = TCO_init_from_boxes_zup_autodepth(model_points_3d=points_init, boxes_2d=pred_bbox, K=K_init)

        if cfg.niter > 0:
            with torch.no_grad():
                outputs = coarse_model(images=images, K=K_init, labels=[model_label],
                                           TCO=TCO_init, n_iterations=1, update_focal_length=True)
            iter_outputs = outputs[f'iteration={1}']
            TCO_coarse = iter_outputs['TCO_output']
            K_init = iter_outputs['K_output']

            with torch.no_grad():
                outputs = refine_model(images=images, K=K_init, labels=[model_label],
                                       TCO=TCO_coarse, n_iterations=cfg.niter, update_focal_length=True)

            iter_outputs = outputs[f'iteration={cfg.niter}']
            TCO_pred = iter_outputs['TCO_output']
            K_pred = iter_outputs['K_output']
        else:
            with torch.no_grad():
                outputs = coarse_model(images=images, K=K_init, labels=[model_label],
                                           TCO=TCO_init, n_iterations=1, update_focal_length=True)

            iter_outputs = outputs[f'iteration={1}']
            TCO_pred = iter_outputs['TCO_output']
            K_pred = iter_outputs['K_output']


        obj_infos = [dict(name=model_label, TWO=np.eye(4))]
        cam_infos = [dict(TWC=np.linalg.inv(TCO_pred[0].cpu().numpy()), resolution=cfg.input_resize, K=K_pred[0].cpu().numpy())]
        rgb_pred_ren = renderer.render_scene(obj_infos, cam_infos)[0]['rgb']
        mask_pred = renderer.render_scene(obj_infos, cam_infos)[0]['mask']

        edges = feature.canny(mask_pred, sigma=0)
        edges = morphology.binary_dilation(edges, selem=np.ones((4, 4)))

        rgb_pred = target_im.copy().transpose(2, 0, 1)
        rgb_pred[0, edges] = 255
        rgb_pred[:, (mask_pred != 255)] = rgb_pred_ren.transpose(2, 0, 1)[:, (mask_pred != 255)]
        image = np.concatenate([target_im.transpose(2, 0, 1), rgb_pred], axis=-1).transpose(1, 2, 0)

        img_name = (cfg.img.split('/')[-1]).split('.')[0]
        cv2.imwrite(f'{img_name}_output_model_{model_id}.jpg',image[:, :, [2, 1, 0]])
        np.savetxt(f'{img_name}_output_model_{model_id}_K.txt', K_pred[0].cpu().numpy())
        np.savetxt(f'{img_name}_output_model_{model_id}_TCO.txt', TCO_pred[0].cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pose evaluation')
    parser.add_argument('--img', default='', type=str)
    parser.add_argument('--cls', default='', type=str,
                        choices=['chair', 'sofa', 'table', 'bed', 'compcars', 'stanfordcars'])
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--topk', default=15, type=int)
    args = parser.parse_args()
    cfg = argparse.ArgumentParser('').parse_args([])

    assign_gpu()

    torch.manual_seed(42)
    np.random.seed(42)


    # Detection Model
    cfg.backbone_detector = 'resnet50-fpn'
    cfg.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    # Pose Model
    cfg.backbone_str = 'resnet50'
    cfg.backbone_pretrained = True
    cfg.n_pose_dims = 9
    cfg.n_dataloader_workers = 8
    cfg.niter = args.niter
    cfg.img = args.img
    cfg.cls = args.cls
    cfg.topk = args.topk

    # Data
    if 'cars' in cfg.cls:
        cfg.input_resize = (200, 300)
    else:
        cfg.input_resize = (640, 640)

    if 'cars' in cfg.cls:
        cfg.n_rendering_workers = 16
    elif 'chair' in cfg.cls:
        cfg.n_rendering_workers = 16
    else:
        cfg.n_rendering_workers = 8

    evaluate(cfg=cfg)