import torch
import numpy as np

from focalpose.lib3d.focalpose_ops import TCO_init_from_boxes, l1, TCO_init_from_boxes_zup_autodepth
from focalpose.lib3d.camera_geometry import project_points_robust as project_points
from focalpose.lib3d.transform_ops import add_noise, add_noise_f
from focalpose.lib3d.focalpose_ops import (
    loss_refiner_CO_disentangled,
    loss_refiner_CO_disentangled_quaternions,
)
from focalpose.lib3d.mesh_losses import compute_ADD_L1_loss
from focalpose.datasets.utils import perturb_detections


def cast(obj):
    return obj.cuda(non_blocking=True)


def h_pose(model, mesh_db, data, meters,
           cfg, n_iterations=1, input_generator='fixed'):

    batch_size, _, h, w = data.images.shape

    images = cast(data.images).float() / 255.
    K_gt = cast(data.K).float()
    TCO_gt = cast(data.TCO).float()
    labels = np.array([obj['name'] for obj in data.objects])

    if cfg.perturb_bboxes:
        bboxes = cast(perturb_detections(data)).float()
    else:
        bboxes = cast(data.bboxes).float()

    meshes = mesh_db.select(labels)
    points = meshes.sample_points(cfg.n_points_loss, deterministic=False)
    TCO_possible_gt = TCO_gt.unsqueeze(1) @ meshes.symmetries

    if input_generator == 'fixed':
        if cfg.predict_focal_length:
            K_init = K_gt.clone()
            K_init[:, 0, 0] = 600
            K_init[:, 1, 1] = 600
            TCO_init = TCO_init_from_boxes_zup_autodepth(model_points_3d=points, boxes_2d=bboxes, K=K_init)
        else:
            TCO_init = TCO_init_from_boxes_zup_autodepth(model_points_3d=points, boxes_2d=bboxes, K=K_gt)
            K_init = K_gt
    elif input_generator == 'gt+noise':
        TCO_init = add_noise(TCO_possible_gt[:, 0], euler_deg_std=[15, 15, 15], trans_std=[0.01, 0.01, 0.05])
        K_init = add_noise_f(K_gt, f_std_frac=0.15)
    else:
        raise ValueError('Unknown input generator', input_generator)

    # model.module.enable_debug()
    outputs = model(images=images, K=K_init, labels=labels,
                    TCO=TCO_init, n_iterations=n_iterations, update_focal_length=cfg.predict_focal_length)
    # raise ValueError

    losses_TCO_iter = []
    losses_f_iter = []
    for n in range(n_iterations):
        iter_outputs = outputs[f'iteration={n+1}']
        K_crop = iter_outputs['K_crop_input']
        TCO_input = iter_outputs['TCO_input']
        TCO_pred = iter_outputs['TCO_output']
        model_outputs = iter_outputs['model_outputs']

        if cfg.loss_disentangled:
            if cfg.n_pose_dims == 9:
                loss_fn = loss_refiner_CO_disentangled
            elif cfg.n_pose_dims == 7:
                loss_fn = loss_refiner_CO_disentangled_quaternions
            else:
                raise ValueError
            pose_outputs = model_outputs['pose']
            loss_TCO_iter = loss_fn(
                TCO_possible_gt=TCO_possible_gt,
                TCO_input=TCO_input,
                refiner_outputs=pose_outputs,
                K_crop=K_crop, points=points,
            )
        else:
            loss_TCO_iter = compute_ADD_L1_loss(
                TCO_possible_gt[:, 0], TCO_pred, points
            )

        meters[f'loss_TCO-iter={n+1}'].add(loss_TCO_iter.mean().item())
        losses_TCO_iter.append(loss_TCO_iter)

        if cfg.predict_focal_length:
            TCO_pred = iter_outputs['TCO_output']
            K_pred = iter_outputs['K_output']
            if cfg.loss_f_type == 'huber':
                t = l1(torch.log(K_pred[:, 0, 0]) - torch.log(K_gt[:, 0, 0]).unsqueeze(dim=-1))
                loss_f_iter = torch.where(t < 1, 0.5 * t ** 2, t - 0.5).mean(dim=-1) * cfg.loss_f_lambda
            elif cfg.loss_f_type == 'reprojection':
                pts_pred_proj = project_points(points, K_pred, TCO_pred)
                pts_gt_proj = project_points(points, K_gt, TCO_possible_gt[:, 0])
                loss_f_iter = l1(pts_pred_proj - pts_gt_proj).flatten(-2, -1).mean(dim=-1) * cfg.loss_f_lambda
            elif cfg.loss_f_type == 'huber+reprojection':
                t = l1(torch.log(K_pred[:, 0, 0]) - torch.log(K_gt[:, 0, 0]).unsqueeze(dim=-1))
                loss_f_huber = torch.where(t < 1, 0.5 * t ** 2, t - 0.5).mean(dim=-1)
                pts_pred_proj = project_points(points, K_pred, TCO_pred)
                pts_gt_proj = project_points(points, K_gt, TCO_possible_gt[:, 0])
                loss_f_proj = l1(pts_pred_proj - pts_gt_proj).flatten(-2, -1).mean(dim=-1)
                loss_f_iter = (loss_f_huber + loss_f_proj) * cfg.loss_f_lambda
            elif cfg.loss_f_type == 'huber+reprojection+disent':
                t = l1(torch.log(K_pred[:, 0, 0]) - torch.log(K_gt[:, 0, 0]).unsqueeze(dim=-1))
                loss_f_huber = torch.where(t < 1, 0.5 * t ** 2, t - 0.5).mean(dim=-1)
                pts_pred_proj_1 = project_points(points, K_gt, TCO_pred)
                pts_pred_proj_2 = project_points(points, K_pred, TCO_possible_gt[:, 0])
                pts_gt_proj = project_points(points, K_gt, TCO_possible_gt[:, 0])
                loss_f_proj = 0.5*(l1(pts_pred_proj_1 - pts_gt_proj).flatten(-2, -1).mean(dim=-1) + l1(pts_pred_proj_2 - pts_gt_proj).flatten(-2, -1).mean(dim=-1))
                loss_f_iter = (loss_f_huber + loss_f_proj) * cfg.loss_f_lambda
            else:
                raise ValueError('Unsupported', cfg.loss_f_type)
            meters[f'loss_f-iter={n+1}'].add(loss_f_iter.mean().item())
            losses_f_iter.append(loss_f_iter)

    loss_TCO = torch.cat(losses_TCO_iter).mean()
    loss = loss_TCO
    meters['loss_TCO'].add(loss_TCO.item())

    if cfg.predict_focal_length:
        loss_f = torch.cat(losses_f_iter).mean()
        meters['loss_f'].add(loss_f.item())
        loss += loss_f
        loss /= 2

    meters['loss_total'].add(loss.item())
    return loss
