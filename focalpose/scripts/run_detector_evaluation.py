import argparse
import torch

from focalpose.utils.resources import assign_gpu
from focalpose.models.mask_rcnn import DetectorMaskRCNN
from focalpose.evaluation.detector_evaluator import evaluate

from torch.utils.data import DataLoader, ConcatDataset
from focalpose.config import EXP_DIR
from focalpose.datasets.datasets_cfg import make_scene_dataset
from focalpose.datasets.detection_dataset import DetectionDataset
from focalpose.utils.logging import get_logger
from torch.backends import cudnn

cudnn.benchmark = True
logger = get_logger(__name__)

def make_datasets(dataset_names):
    datasets = []
    all_labels = set()
    for (ds_name, n_repeat) in dataset_names:
        ds = make_scene_dataset(ds_name)
        all_labels = all_labels.union(set(ds.all_labels))
        for _ in range(n_repeat):
            datasets.append(ds)
    return ConcatDataset(datasets), all_labels

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pose evaluation')
    parser.add_argument('--model-run-id', default='', type=str)
    parser.add_argument('--dataset', default='', type=str)
    args = parser.parse_args()
    cfg = argparse.ArgumentParser('').parse_args([])

    # Data
    cfg.dataset = args.dataset
    cfg.input_resize = (480, 640)

    # Model
    cfg.backbone_str = 'resnet50-fpn'
    cfg.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    pth_dir = EXP_DIR / args.model_run_id
    path = pth_dir / 'checkpoint.pth.tar'

    assign_gpu()

    scene_ds_val, labels = make_datasets([(f'{cfg.dataset}.val', 1)])
    label_to_category_id = dict()
    label_to_category_id['background'] = 0
    for n, label in enumerate(sorted(list(labels)), 1):
        label_to_category_id[label] = n

    ds_kwargs = dict(
        resize=cfg.input_resize,
        rgb_augmentation=False,
        background_augmentation=False,
        gray_augmentation=False,
        label_to_category_id=label_to_category_id,
    )
    ds_val = DetectionDataset(scene_ds_val, **ds_kwargs)
    ds_iter_val = DataLoader(ds_val, batch_size=1,
                             num_workers=1,
                             collate_fn=collate_fn,
                             drop_last=False, shuffle=False, pin_memory=True)

    model = DetectorMaskRCNN(input_resize=cfg.input_resize,
                             n_classes=len(label_to_category_id),
                             backbone_str='resnet50-fpn',
                             anchor_sizes=cfg.anchor_sizes).cuda()

    model.load_state_dict(torch.load(path.as_posix())['state_dict'])
    evaluate(model, ds_iter_val, device=torch.device('cuda'))


