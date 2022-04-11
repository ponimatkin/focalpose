import yaml
import numpy as np
import time
import torch
import simplejson as json
from tqdm import tqdm
import functools
from pathlib import Path
from torchnet.meter import AverageValueMeter
from collections import defaultdict
import torch.distributed as dist

from focalpose.config import EXP_DIR

from torch.utils.data import DataLoader, ConcatDataset
from focalpose.utils.multiepoch_dataloader import MultiEpochDataLoader

from focalpose.datasets.datasets_cfg import make_scene_dataset, make_urdf_dataset
from focalpose.datasets.pose_dataset import PoseDataset
from focalpose.datasets.samplers import PartialSampler

# Evaluation
from focalpose.evaluation.pose_evaluator import pose_evaluator

from focalpose.rendering.bullet_batch_renderer import BulletBatchRenderer
from focalpose.lib3d.rigid_mesh_database import MeshDataBase

from .pose_forward_loss import h_pose
from .pose_models_cfg import create_model_pose


from focalpose.utils.logging import get_logger
from focalpose.utils.distributed import get_world_size, get_rank, sync_model, init_distributed_mode, reduce_dict
from torch.backends import cudnn

cudnn.benchmark = True
logger = get_logger(__name__)

def log(config, model,
        log_dict, test_dict, epoch, best_f, best_loss):
    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)
    log_dict.update(epoch=epoch)
    if not (save_dir / 'config.yaml').exists():
        (save_dir / 'config.yaml').write_text(yaml.dump(config))

    def save_checkpoint(model, best_f, best_loss):
        if best_f:
            ckpt_name = 'best_f_checkpoint'
        elif best_loss:
            ckpt_name = 'best_loss_checkpoint'
        else:
            ckpt_name = 'checkpoint'
        ckpt_name += '.pth.tar'
        path = save_dir / ckpt_name
        torch.save({'state_dict': model.module.state_dict(),
                    'epoch': epoch}, path)

    save_checkpoint(model, best_f=False, best_loss=False)
    if best_f:
        save_checkpoint(model, best_f=True, best_loss=False)

    if best_loss:
        save_checkpoint(model, best_f=False, best_loss=True)

    with open(save_dir / 'log.txt', 'a') as f:
        f.write(json.dumps(log_dict, ignore_nan=True) + '\n')

    if test_dict is not None:
        for ds_name, ds_errors in test_dict.items():
            ds_errors['epoch'] = epoch
            with open(save_dir / f'errors_{ds_name}.txt', 'a') as f:
                f.write(json.dumps(test_dict[ds_name], ignore_nan=True) + '\n')

    logger.info(config.run_id)
    logger.info(log_dict)
    logger.info(test_dict)


def train_pose(args):
    torch.set_num_threads(1)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / 'config.yaml').read_text())
        keep_fields = set(['resume_run_id', 'epoch_size', ])
        vars(args).update({k: v for k, v in vars(resume_args).items() if k not in keep_fields})

    args.train_refiner = args.TCO_input_generator == 'gt+noise'
    args.train_coarse = not args.train_refiner
    args.save_dir = EXP_DIR / args.run_id

    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    # Initialize distributed
    device = torch.cuda.current_device()
    init_distributed_mode()
    world_size = get_world_size()
    args.n_gpus = world_size
    args.global_batch_size = world_size * args.batch_size
    logger.info(f'Connection established with {world_size} gpus.')

    def make_datasets(dataset_names):
        train_datasets = []
        val_datasets = []
        for (ds_name, n_repeat, n_frames) in dataset_names:
            if 'synthetic.' not in ds_name:
                ds_train = make_scene_dataset(ds_name)
                ds_val = make_scene_dataset(ds_name)

                ds_train.index = ds_train.index.sample(frac=0.95, random_state=21022021)
                ds_val.index = ds_val.index.drop(ds_train.index.index)

                ds_train.index = ds_train.index.reset_index(drop=True)
                ds_val.index = ds_val.index.reset_index(drop=True)

                val_datasets.append(ds_val)

                for _ in range(n_repeat):
                    train_datasets.append(ds_train)

                logger.info(f'Loaded {ds_name} (train) with {len(ds_train) * n_repeat} images.')
                logger.info(f'Loaded {ds_name} (val) with {len(ds_val)} images.')
            else:
                ds_train = make_scene_dataset(ds_name)
                train_datasets.append(ds_train)
                logger.info(f'Loaded {ds_name} (train) with {len(ds_train) * n_repeat} images.')

        return ConcatDataset(train_datasets), ConcatDataset(val_datasets)

    def make_test_datasets(dataset_names):
        test_datasets = []

        for (ds_name, n_repeat, n_frames) in dataset_names:
            ds_test = make_scene_dataset(ds_name)
            test_datasets.append(ds_test)
            logger.info(f'Loaded {ds_name} (test) with {len(ds_test) * n_repeat} images.')

        return ConcatDataset(test_datasets)

    scene_ds_train, scene_ds_val = make_datasets(args.train_ds_names)
    scene_ds_test = make_test_datasets(args.test_ds_names)

    ds_kwargs = dict(
        resize=args.input_resize,
        rgb_augmentation=args.rgb_augmentation,
        background_augmentation=args.background_augmentation,
        min_area=args.min_area,
        gray_augmentation=args.gray_augmentation,
    )

    ds_kwargs_eval = dict(
        resize=args.input_resize,
        rgb_augmentation=False,
        background_augmentation=False,
        min_area=None,
        gray_augmentation=False
    )

    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = PoseDataset(scene_ds_val, **ds_kwargs_eval)
    ds_test = PoseDataset(scene_ds_test, **ds_kwargs_eval)

    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(ds_train, sampler=train_sampler, batch_size=args.batch_size,
                               num_workers=args.n_dataloader_workers, collate_fn=ds_train.collate_fn,
                               drop_last=False, pin_memory=True)
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    val_sampler = PartialSampler(ds_val, epoch_size=len(ds_val))
    ds_iter_val = DataLoader(ds_val, sampler=val_sampler, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers, collate_fn=ds_val.collate_fn,
                             drop_last=False, pin_memory=True)
    ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    test_sampler = PartialSampler(ds_test, epoch_size=len(ds_test))
    ds_iter_test = DataLoader(ds_test, sampler=test_sampler, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers, collate_fn=ds_test.collate_fn,
                             drop_last=False, pin_memory=True)
    ds_iter_test = MultiEpochDataLoader(ds_iter_test)

    # Make model
    renderer = BulletBatchRenderer(object_set=args.urdf_ds_name, n_workers=args.n_rendering_workers, preload_cache=False, split_objects=True)
    object_ds = make_urdf_dataset(args.urdf_ds_name)
    mesh_db = MeshDataBase.from_urdf_ds(object_ds).batched(n_sym=1, resample_n_points=20000).cuda().float()

    model = create_model_pose(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        path = resume_dir / 'checkpoint.pth.tar'
        logger.info(f'Loading checkpoing from {path}')
        save = torch.load(path)
        state_dict = save['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = save['epoch'] + 1
    else:
        start_epoch = 0
    end_epoch = args.n_epochs

    if args.run_id_pretrain is not None:
        pretrain_path = EXP_DIR / args.run_id_pretrain / 'checkpoint.pth.tar'
        logger.info(f'Using pretrained model from {pretrain_path}.')
        model.load_state_dict(torch.load(pretrain_path)['state_dict'])

    # Synchronize models across processes.
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup
    def get_lr_ratio(batch):
        n_batch_per_epoch = args.epoch_size // args.batch_size
        epoch_id = batch // n_batch_per_epoch

        if args.n_epochs_warmup == 0:
            lr_ratio = 1.0
        else:
            n_batches_warmup = args.n_epochs_warmup * (args.epoch_size // args.batch_size)
            lr_ratio = min(max(batch, 1) / n_batches_warmup, 1.0)

        lr_ratio /= 10 ** (epoch_id // args.lr_epoch_decay)
        return lr_ratio

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_ratio)
    lr_scheduler.last_epoch = start_epoch * args.epoch_size // args.batch_size - 1

    # Just remove the annoying warning
    optimizer._step_count = 1
    lr_scheduler.step()
    optimizer._step_count = 0

    if get_rank() == 0:
        best_val_f = np.inf
        best_val_loss = np.inf
    for epoch in range(start_epoch, end_epoch):
        meters_train = defaultdict(lambda: AverageValueMeter())
        meters_val = defaultdict(lambda: AverageValueMeter())
        meters_test = defaultdict(lambda: AverageValueMeter())
        meters_time = defaultdict(lambda: AverageValueMeter())

        h = functools.partial(h_pose, model=model, cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator)
        e = functools.partial(pose_evaluator, model=model, cfg=args, n_iterations=1,
                                    mesh_db=mesh_db, input_generator=args.TCO_input_generator)

        def train_epoch():
            model.train()
            iterator = tqdm(ds_iter_train, ncols=80)
            t = time.time()
            for n, sample in enumerate(iterator):
                if n > 0:
                    meters_time['data'].add(time.time() - t)

                optimizer.zero_grad()

                t = time.time()
                loss = h(data=sample, meters=meters_train)
                meters_time['forward'].add(time.time() - t)
                iterator.set_postfix(loss=loss.item())
                meters_train['loss_total'].add(loss.item())

                t = time.time()
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)
                meters_train['grad_norm'].add(torch.as_tensor(total_grad_norm).item())

                optimizer.step()
                meters_time['backward'].add(time.time() - t)
                meters_time['memory'].add(torch.cuda.max_memory_allocated() / 1024. ** 2)

                lr_scheduler.step()
                t = time.time()

        @torch.no_grad()
        def validation():
            model.eval()
            for sample in tqdm(ds_iter_val, ncols=80):
                loss = h(data=sample, meters=meters_val)
                e(data=sample, meters=meters_val)
                meters_val['loss_total'].add(loss.item())

        @torch.no_grad()
        def test():
            model.eval()
            for sample in tqdm(ds_iter_test, ncols=80):
                loss = h(data=sample, meters=meters_test)
                e(data=sample, meters=meters_test)
                meters_test['loss_total'].add(loss.item())

        train_epoch()
        if epoch % args.eval_epoch_interval == 0:
            validation()
            test()

        log_dict = dict()
        log_dict.update({
            'grad_norm': meters_train['grad_norm'].mean,
            'grad_norm_std': meters_train['grad_norm'].std,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time_forward': meters_time['forward'].mean,
            'time_backward': meters_time['backward'].mean,
            'time_data': meters_time['data'].mean,
            'gpu_memory': meters_time['memory'].mean,
            'time': time.time(),
            'n_iterations': (epoch + 1) * len(ds_iter_train),
            'n_datas': (epoch + 1) * args.global_batch_size * len(ds_iter_train),
        })

        for string, meters in zip(('train', 'val', 'test'), (meters_train, meters_val, meters_test)):
            for k in dict(meters).keys():
                log_dict[f'{string}_{k}'] = meters[k].mean

        log_dict = reduce_dict(log_dict)
        if get_rank() == 0:
            if epoch % args.eval_epoch_interval == 0:
                if log_dict['val_f_err_median'] < best_val_f:
                    best_val_f = log_dict['val_f_err_median']
                    best_f = True
                else:
                    best_f = False

                if log_dict['val_loss_total'] < best_val_loss:
                    best_val_loss = log_dict['val_loss_total']
                    best_loss = True
                else:
                    best_loss = False

            else:
                best_f = False
                best_loss = False

            log(config=args, model=model, epoch=epoch, log_dict=log_dict, test_dict=None, best_f=best_f, best_loss=best_loss)
        dist.barrier()
