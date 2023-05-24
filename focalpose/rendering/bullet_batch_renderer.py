import torch
import numpy as np
import multiprocessing

from focalpose.lib3d.transform_ops import invert_T
from focalpose.datasets.datasets_cfg import make_urdf_dataset
from .bullet_scene_renderer import BulletSceneRenderer


def init_renderer(urdf_ds, preload=True):
    renderer = BulletSceneRenderer(urdf_ds=urdf_ds,
                                   preload_cache=preload,
                                   background_color=(0, 0, 0))
    return renderer


def worker_loop(worker_id, in_queue, out_queue, object_set, preload=True):
    renderer = init_renderer(object_set, preload=preload)
    while True:
        kwargs = in_queue.get()
        if kwargs is None:
            return
        obj_infos = kwargs['obj_infos']
        cam_infos = kwargs['cam_infos']
        render_depth = kwargs['render_depth']
        is_valid = np.isfinite(obj_infos[0]['TWO']).all() \
            and np.isfinite(cam_infos[0]['TWC']).all() \
            and np.isfinite(cam_infos[0]['K']).all()
        if is_valid:
            cam_obs = renderer.render_scene(cam_infos=cam_infos, obj_infos=obj_infos,
                                            render_depth=render_depth)
            images = np.stack([d['rgb'] for d in cam_obs])
            depth = np.stack([d['depth'] for d in cam_obs]) if render_depth else None
        else:
            res = cam_infos[0]['resolution']
            images = np.zeros((1, min(res), max(res), 3), dtype=np.uint8)
            depth = np.zeros((1, min(res), max(res)), dtype=np.float32)
        out_queue.put((kwargs['data_id'], images, depth))


class BulletBatchRenderer:
    def __init__(self, object_set,
                 n_workers=8,
                 preload_cache=True,
                 split_objects=True):

        self.split_objects = split_objects
        self.urdf_ds = make_urdf_dataset(object_set)
        self.object_labels = self.urdf_ds.index['label'].tolist()
        if self.split_objects:
            assert not preload_cache, 'Preload cache means loading all objects in all processes which is incompatbiel with split_objects'
        # self.object_label_to_target_workers = dict()
        # object_labels =
        # object_labels_split = np.array_split(object_labels, n_workers)

        # if self.split_objects:
        #     for n in range(n_workers):
        #         self.object_label_to_target_workers =

        # for idx, row in self.urdf_ds.index.iteritems():
        #     if self.split_objects:
        #         self.object_label_to_target_workers = object_label_to_target_workers
        #     else:
        #         self.object_label_to_target_workers = set(list(range(n_workers)))
        self.object_set = object_set
        self.n_workers = n_workers
        self.init_plotters(preload_cache)

    def render(self, obj_infos, TCO, K, resolution=(240, 320), render_depth=False):
        TCO = torch.as_tensor(TCO).detach()
        TOC = invert_T(TCO).cpu().numpy()
        K = torch.as_tensor(K).cpu().numpy()
        bsz = len(TCO)
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)

        # NOTE: Could be faster with pytorch 3.8's sharedmemory
        for n in np.arange(bsz):
            obj_info = dict(
                name=obj_infos[n]['name'],
                TWO=np.eye(4)
            )
            cam_info = dict(
                resolution=resolution,
                K=K[n],
                TWC=TOC[n],
            )
            kwargs = dict(cam_infos=[cam_info], obj_infos=[obj_info], render_depth=render_depth)
            if self.n_workers > 0:
                kwargs['data_id'] = n
                in_queue = self.object_label_to_queue[obj_info['name']]
                in_queue.put(kwargs)
            else:
                cam_obs = self.plotters[0].render_scene(**kwargs)
                images = np.stack([d['rgb'] for d in cam_obs])
                depth = np.stack([d['depth'] for d in cam_obs]) if render_depth else None
                self.out_queue.put((n, images, depth))

        images = [None for _ in np.arange(bsz)]
        depths = [None for _ in np.arange(bsz)]
        for n in np.arange(bsz):
            data_id, im, depth = self.out_queue.get()
            images[data_id] = im[0]
            if render_depth:
                depths[data_id] = depth[0]
        images = torch.as_tensor(np.stack(images, axis=0)).pin_memory().cuda(non_blocking=True)
        images = images.float().permute(0, 3, 1, 2) / 255

        if render_depth:
            depths = torch.as_tensor(np.stack(depths, axis=0)).pin_memory().cuda(non_blocking=True)
            depths = depths.float()
            return images, depths
        else:
            return images

    def init_plotters(self, preload_cache):
        self.plotters = []
        if self.split_objects:
            self.in_queues = [multiprocessing.Queue() for _ in range(self.n_workers)]
            object_labels_split = np.array_split(self.object_labels, self.n_workers)
            self.worker_id_to_queue = {n: self.in_queues[n] for n in range(self.n_workers)}
            self.object_label_to_queue = dict()
            for n, split in enumerate(object_labels_split):
                for l in split:
                    self.object_label_to_queue[l] = self.in_queues[n]
        else:
            self.in_queues = [multiprocessing.Queue()]
            self.object_label_to_queue = {k: self.in_queues[0] for k in self.object_labels}
            self.worker_id_to_queue = {n: self.in_queues[0] for n in range(self.n_workers)}

        self.out_queue = multiprocessing.Queue()

        if self.n_workers > 0:
            for n in range(self.n_workers):
                plotter = multiprocessing.Process(target=worker_loop,
                                                  kwargs=dict(worker_id=n,
                                                              in_queue=self.worker_id_to_queue[n],
                                                              out_queue=self.out_queue,
                                                              preload=preload_cache,
                                                              object_set=self.object_set))
                plotter.start()
                self.plotters.append(plotter)
        else:
            self.plotters = [init_renderer(self.object_set, preload_cache)]

    def stop(self):
        if self.n_workers > 0:
            for n in range(self.n_workers):
                self.worker_id_to_queue[n].put(None)
            for p in self.plotters:
                p.join()
                p.terminate()
        for queue in self.in_queues:
            queue.close()
        self.out_queue.close()

    def __del__(self):
        self.stop()
