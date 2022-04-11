import argparse
import numpy as np
import os
from colorama import Fore, Style
from focalpose.utils.resources import assign_gpu

from focalpose.training.train_detector import train_detector
from focalpose.utils.logging import get_logger
logger = get_logger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    assign_gpu()

    cfg = argparse.ArgumentParser('').parse_args([])
    if args.config:
        logger.info(f"{Fore.GREEN}Training with config: {args.config} {Style.RESET_ALL}")

    cfg.resume_run_id = None
    if len(args.resume) > 0:
        cfg.resume_run_id = args.resume
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")

    N_CPUS = int(os.environ.get('N_CPUS', 10))
    N_GPUS = int(os.environ.get('N_PROCS', 1))
    N_WORKERS = min(N_CPUS - 2, 8)
    N_RAND = np.random.randint(1e6)
    cfg.n_gpus = N_GPUS

    run_comment = ''

    if 'pix3d-sofa' in args.config:
        cfg.urdf_ds_name = 'pix3d-sofa'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'F005p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1), ('pix3d-sofa.train', 1)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1), ('pix3d-sofa.train', 10)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'F1p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1), ('pix3d-sofa.train', 20)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'F2p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1), ('pix3d-sofa.train', 40)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1), ('pix3d-sofa.train', 200)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'F20p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1), ('pix3d-sofa.train', 400)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-sofa.train', 1)]
            cfg.val_ds_names = [('pix3d-sofa.test', 1)]

    elif 'pix3d-bed' in args.config:
        cfg.urdf_ds_name = 'pix3d-bed'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'F005p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1), ('pix3d-bed.train', 1)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1), ('pix3d-bed.train', 25)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'F1p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1), ('pix3d-bed.train', 50)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'F2p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1), ('pix3d-bed.train', 100)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1), ('pix3d-bed.train', 500)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'F20p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1), ('pix3d-bed.train', 1000)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-bed.train', 1)]
            cfg.val_ds_names = [('pix3d-bed.test', 1)]

    elif 'pix3d-table' in args.config:
        cfg.urdf_ds_name = 'pix3d-table'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'F005p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1), ('pix3d-table.train', 1)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1), ('pix3d-table.train', 20)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'F1p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1), ('pix3d-table.train', 40)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'F2p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1), ('pix3d-table.train', 80)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1), ('pix3d-table.train', 400)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'F20p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1), ('pix3d-table.train', 800)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-table.train', 1)]
            cfg.val_ds_names = [('pix3d-table.test', 1)]

    elif 'pix3d-chair' in args.config:
        cfg.urdf_ds_name = 'pix3d-chair'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-chair-1-1M.train', 1),
                                  ('synthetic.pix3d-chair-2-1M.train', 1),
                                  ('synthetic.pix3d-chair-3-1M.train', 1),
                                  ('synthetic.pix3d-chair-4-1M.train', 1),
                                  ('synthetic.pix3d-chair-5-1M.train', 1),
                                  ('synthetic.pix3d-chair-6-1M.train', 1),
                                  ('synthetic.pix3d-chair-7-1M.train', 1),
                                  ('synthetic.pix3d-chair-8-1M.train', 1),
                                  ('synthetic.pix3d-chair-9-1M.train', 1)]
            cfg.val_ds_names = [('pix3d-chair.test', 1)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-chair-1-1M.train', 1),
                                  ('synthetic.pix3d-chair-2-1M.train', 1),
                                  ('synthetic.pix3d-chair-3-1M.train', 1),
                                  ('synthetic.pix3d-chair-4-1M.train', 1),
                                  ('synthetic.pix3d-chair-5-1M.train', 1),
                                  ('synthetic.pix3d-chair-6-1M.train', 1),
                                  ('synthetic.pix3d-chair-7-1M.train', 1),
                                  ('synthetic.pix3d-chair-8-1M.train', 1),
                                  ('synthetic.pix3d-chair-9-1M.train', 1),
                                  ('pix3d-chair.train', 6)]
            cfg.val_ds_names = [('pix3d-chair.test', 1)]
        elif 'F20p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-chair-1-1M.train', 1),
                                  ('synthetic.pix3d-chair-2-1M.train', 1),
                                  ('synthetic.pix3d-chair-3-1M.train', 1),
                                  ('synthetic.pix3d-chair-4-1M.train', 1),
                                  ('synthetic.pix3d-chair-5-1M.train', 1),
                                  ('synthetic.pix3d-chair-6-1M.train', 1),
                                  ('synthetic.pix3d-chair-7-1M.train', 1),
                                  ('synthetic.pix3d-chair-8-1M.train', 1),
                                  ('synthetic.pix3d-chair-9-1M.train', 1),
                                  ('pix3d-chair.train', 12)]
            cfg.val_ds_names = [('pix3d-chair.test', 1)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-chair.train', 1)]
            cfg.val_ds_names = [('pix3d-chair.test', 1)]

    elif 'pix3d-agnostic' in args.config:
        if 'real' in args.config:
            cfg.train_ds_names = [
                ('pix3d-chair.train', 1),
                ('pix3d-table.train', 1),
                ('pix3d-sofa.train', 1),
                ('pix3d-bed.train', 1)
            ]
            cfg.val_ds_names = [
                ('pix3d-chair.test', 1),
                ('pix3d-table.test', 1),
                ('pix3d-sofa.test', 1),
                ('pix3d-bed.test', 1)
            ]

    elif 'compcars3d' in args.config:
        cfg.urdf_ds_name = 'compcars3d'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.compcars3d-1-1M.train', 1),
                                  ('synthetic.compcars3d-2-1M.train', 1),
                                  ('synthetic.compcars3d-3-1M.train', 1),
                                  ('synthetic.compcars3d-4-1M.train', 1),
                                  ('synthetic.compcars3d-5-1M.train', 1),
                                  ('synthetic.compcars3d-6-1M.train', 1),
                                  ('synthetic.compcars3d-7-1M.train', 1),
                                  ('synthetic.compcars3d-8-1M.train', 1),
                                  ('synthetic.compcars3d-9-1M.train', 1),
                                  ('synthetic.compcars3d-10-1M.train', 1),
                                  ('compcars3d.train', 3)]
            cfg.val_ds_names = [('compcars3d.test', 1)]
        if 'F20p' in args.config:
            cfg.train_ds_names = [('synthetic.compcars3d-1-1M.train', 1),
                                  ('synthetic.compcars3d-2-1M.train', 1),
                                  ('synthetic.compcars3d-3-1M.train', 1),
                                  ('synthetic.compcars3d-4-1M.train', 1),
                                  ('synthetic.compcars3d-5-1M.train', 1),
                                  ('synthetic.compcars3d-6-1M.train', 1),
                                  ('synthetic.compcars3d-7-1M.train', 1),
                                  ('synthetic.compcars3d-8-1M.train', 1),
                                  ('synthetic.compcars3d-9-1M.train', 1),
                                  ('synthetic.compcars3d-10-1M.train', 1),
                                  ('compcars3d.train', 60)]
            cfg.val_ds_names = [('compcars3d.test', 1)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('compcars3d.train', 1)]
            cfg.val_ds_names = [('compcars3d.test', 1)]

    elif 'stanfordcars3d' in args.config:
        cfg.urdf_ds_name = 'stanfordcars3d'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.stanfordcars3d-1-1M.train', 1),
                                  ('synthetic.stanfordcars3d-2-1M.train', 1),
                                  ('synthetic.stanfordcars3d-3-1M.train', 1),
                                  ('synthetic.stanfordcars3d-4-1M.train', 1),
                                  ('synthetic.stanfordcars3d-5-1M.train', 1),
                                  ('synthetic.stanfordcars3d-6-1M.train', 1),
                                  ('synthetic.stanfordcars3d-7-1M.train', 1),
                                  ('synthetic.stanfordcars3d-8-1M.train', 1),
                                  ('synthetic.stanfordcars3d-9-1M.train', 1),
                                  ('synthetic.stanfordcars3d-10-1M.train', 1),
                                  ('synthetic.stanfordcars3d-11-1M.train', 1),
                                  ('synthetic.stanfordcars3d-12-1M.train', 1),
                                  ('synthetic.stanfordcars3d-13-1M.train', 1),
                                  ('stanfordcars3d.train', 1)]
            cfg.val_ds_names = [('stanfordcars3d.test', 1)]
        elif 'F20p' in args.config:
            cfg.train_ds_names = [('synthetic.stanfordcars3d-1-1M.train', 1),
                                  ('synthetic.stanfordcars3d-2-1M.train', 1),
                                  ('synthetic.stanfordcars3d-3-1M.train', 1),
                                  ('synthetic.stanfordcars3d-4-1M.train', 1),
                                  ('synthetic.stanfordcars3d-5-1M.train', 1),
                                  ('synthetic.stanfordcars3d-6-1M.train', 1),
                                  ('synthetic.stanfordcars3d-7-1M.train', 1),
                                  ('synthetic.stanfordcars3d-8-1M.train', 1),
                                  ('synthetic.stanfordcars3d-9-1M.train', 1),
                                  ('synthetic.stanfordcars3d-10-1M.train', 1),
                                  ('synthetic.stanfordcars3d-11-1M.train', 1),
                                  ('synthetic.stanfordcars3d-12-1M.train', 1),
                                  ('synthetic.stanfordcars3d-13-1M.train', 1),
                                  ('stanfordcars3d.train', 20)]
            cfg.val_ds_names = [('stanfordcars3d.test', 1)]

        elif 'real' in args.config:
            cfg.train_ds_names = [('stanfordcars3d.train', 1)]
            cfg.val_ds_names = [('stanfordcars3d.test', 1)]


    cfg.val_epoch_interval = 10
    cfg.test_epoch_interval = 30
    cfg.n_test_frames = None

    if 'cars' in args.config:
        cfg.input_resize = (200, 300)
    else:
        cfg.input_resize = (640, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'resnet50-fpn'
    cfg.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    # Pretraning
    cfg.run_id_pretrain = None
    cfg.pretrain_coco = True

    # Training
    cfg.batch_size = 2
    cfg.epoch_size = 1000
    cfg.n_epochs = 60
    cfg.lr_epoch_decay = 25
    cfg.n_epochs_warmup = 0
    cfg.n_dataloader_workers = N_WORKERS

    # Optimizer
    cfg.optimizer = 'sgd'
    cfg.lr = (0.02 / 8) * N_GPUS * float(cfg.batch_size / 4)
    cfg.weight_decay = 1e-4
    cfg.momentum = 0.9

    # Method
    cfg.rpn_box_reg_alpha = 1
    cfg.objectness_alpha = 1
    cfg.classifier_alpha = 1
    cfg.mask_alpha = 1
    cfg.box_reg_alpha = 1
    cfg.freeze = False
    if 'two-class' in args.config:
        cfg.two_class = True
        cfg.run_id = f'detector-{args.config}-{run_comment}-{N_RAND}'
    else:
        cfg.two_class = False
        cfg.run_id = f'detector-{args.config}-{run_comment}-{N_RAND}'

    if args.debug:
        cfg.n_epochs = 4
        cfg.val_epoch_interval = 1
        cfg.batch_size = 2
        cfg.epoch_size = 10 * cfg.batch_size
        cfg.run_id = 'debug-' + cfg.run_id
        cfg.background_augmentation = False
        cfg.rgb_augmentation = False
        cfg.n_dataloader_workers = 1
        cfg.n_test_frames = 10

    N_GPUS = int(os.environ.get('N_PROCS', 1))
    cfg.epoch_size = cfg.epoch_size // N_GPUS

    train_detector(cfg)
