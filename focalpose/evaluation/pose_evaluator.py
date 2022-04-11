import torch
import numpy as np
from scipy.linalg import logm
from focalpose.lib3d.camera_geometry import project_points_robust as project_points
from focalpose.lib3d.focalpose_ops import TCO_init_from_boxes_zup_autodepth
from focalpose.lib3d.transform_ops import add_noise, add_noise_f, transform_pts


def cast(obj):
    return obj.cuda(non_blocking=True)


def pose_evaluator(model, data, meters, cfg, n_iterations=1, mesh_db=None, input_generator='fixed'):

    batch_size, _, h, w = data.images.shape
    images = cast(data.images).float() / 255.
    K_gt = cast(data.K).float()
    TCO_gt = cast(data.TCO).float()
    labels = np.array([obj['name'] for obj in data.objects])
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

    iter_outputs = outputs[f'iteration={n_iterations}']
    TCO_pred = iter_outputs['TCO_output']
    K_pred = iter_outputs['K_output']

    pts_pred_tr = transform_pts(TCO_pred, meshes.points)
    pts_gt_tr = transform_pts(TCO_gt, meshes.points)
    pts_pred_proj = project_points(points, K_pred, TCO_pred)
    pts_gt_proj = project_points(points, K_gt, TCO_gt)

    er = []
    et = []
    ert = []
    ep = []
    ef = np.abs(K_gt[:, 0, 0].cpu().numpy() - K_pred[:, 0, 0].cpu().numpy())/np.abs(K_gt[:, 0, 0].cpu().numpy())

    for i in range(batch_size):
        R_pred = TCO_pred[i, :3, :3].cpu()
        t_pred = TCO_pred[i, :3, 3].cpu().numpy()
        R_gt = TCO_gt[i, :3, :3].cpu()
        t_gt = TCO_gt[i, :3, 3].cpu().numpy()
        bbox = bboxes[i].cpu().numpy()

        d_img = np.sqrt(w**2 + h**2)
        d_bbox = np.sqrt((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2)
        if np.isclose(d_bbox, 0):
            continue
        er.append(np.linalg.norm(logm(R_gt.numpy().T @ R_pred.numpy(), disp=False)[0], ord='fro') / np.sqrt(2))
        et.append(np.linalg.norm(t_gt - t_pred)/np.linalg.norm(t_gt))
        ert.append((d_bbox/d_img*torch.norm(pts_pred_tr[i] - pts_gt_tr[i], dim=-1).mean().cpu().numpy()/np.linalg.norm(t_gt)))
        ep.append(torch.norm(pts_pred_proj[i] - pts_gt_proj[i], dim=-1).mean().cpu().numpy()/d_bbox)

    meters['R_err_median'].add(np.median(er))
    meters['t_err_median'].add(np.median(et))
    meters['f_err_median'].add(np.median(ef))
    meters['R_acc_30_deg'].add(np.sum(np.array(er) < np.radians(30))/batch_size)
    meters['Rt_err_median'].add(np.median(ert))
    meters['proj_err_median'].add(np.median(ep))
    meters['proj_acc_0.1'].add(np.sum(np.array(ep) < 0.1)/batch_size)




