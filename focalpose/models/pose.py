import torch
from torch import nn

from focalpose.config import DEBUG_DATA_DIR
from focalpose.lib3d.camera_geometry import get_K_crop_resize, boxes_from_uv

from focalpose.lib3d.cropping import deepim_crops_robust as deepim_crops
from focalpose.lib3d.camera_geometry import project_points_robust as project_points

from focalpose.lib3d.rotations import (
    compute_rotation_matrix_from_ortho6d, compute_rotation_matrix_from_quaternions)
from focalpose.lib3d.focalpose_ops import apply_imagespace_predictions

from focalpose.utils.logging import get_logger
logger = get_logger(__name__)


class PosePredictor(nn.Module):
    def __init__(self, backbone, renderer,
                 mesh_db, render_size=(240, 320),
                 pose_dim=9):
        super().__init__()

        self.backbone = backbone
        self.renderer = renderer
        self.mesh_db = mesh_db
        self.render_size = render_size
        self.pose_dim = pose_dim

        n_features = backbone.n_features

        self.heads = dict()
        self.pose_fc = nn.Linear(n_features, pose_dim, bias=True)
        self.heads['pose'] = self.pose_fc

        self.focal_length_fc = nn.Linear(n_features, 1, bias=True)
        self.heads['focal_length'] = self.focal_length_fc

        self.debug = False
        self.tmp_debug = dict()

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False

    def crop_inputs(self, images, K, TCO, labels):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz
        meshes = self.mesh_db.select(labels)
        points = meshes.sample_points(2000, deterministic=True)
        uv = project_points(points, K, TCO)
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, images_cropped = deepim_crops(
            images=images, obs_boxes=boxes_rend, K=K,
            TCO_pred=TCO, O_vertices=points, output_size=self.render_size, lamb=1.4
        )
        K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                   orig_size=images.shape[-2:], crop_resize=self.render_size)
        if self.debug:
            self.tmp_debug.update(
                boxes_rend=boxes_rend,
                rend_center_uv=project_points(torch.zeros(bsz, 1, 3).to(K.device), K, TCO),
                uv=uv,
                boxes_crop=boxes_crop,
            )
        return images_cropped, K_crop, boxes_rend, boxes_crop

    def update_pose(self, TCO, K_crop, pose_outputs):
        if self.pose_dim == 9:
            dR = compute_rotation_matrix_from_ortho6d(pose_outputs[:, 0:6])
            vxvyvz = pose_outputs[:, 6:9]
        elif self.pose_dim == 7:
            dR = compute_rotation_matrix_from_quaternions(pose_outputs[:, 0:4])
            vxvyvz = pose_outputs[:, 4:7]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated

    @staticmethod
    def update_camera_matrix(K, focal_length_update):
        K_updated = K.clone()
        fx = torch.exp(focal_length_update.squeeze(-1))*K[:, 0, 0]
        fy = torch.exp(focal_length_update.squeeze(-1))*K[:, 1, 1]
        K_updated[:, 0, 0] = fx
        K_updated[:, 1, 1] = fy
        return K_updated

    def net_forward(self, x):
        x = self.backbone(x)
        x = x.flatten(2).mean(dim=-1)
        outputs = dict()
        for k, head in self.heads.items():
            outputs[k] = head(x)
        return outputs

    def forward(self, images, K, labels, TCO, n_iterations=1, update_focal_length=False):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz

        outputs = dict()
        TCO_input = TCO
        K_input = K
        for n in range(n_iterations):
            TCO_input = TCO_input.detach()
            K_input = K_input.detach()
            images_crop, K_crop_input, boxes_rend, boxes_crop = self.crop_inputs(images, K_input, TCO_input, labels)
            K_crop_input = K_crop_input.detach()
            renders = self.renderer.render(obj_infos=[dict(name=l) for l in labels],
                                           TCO=TCO_input,
                                           K=K_crop_input, resolution=self.render_size)

            x = torch.cat((images_crop, renders), dim=1)

            model_outputs = self.net_forward(x)

            if update_focal_length:
                K_output = self.update_camera_matrix(K_input, model_outputs['focal_length'])
                _, K_crop_output, _, _ = self.crop_inputs(images, K_output, TCO_input, labels)
            else:
                K_output = K_input
                K_crop_output = K_crop_input

            TCO_output = self.update_pose(TCO_input, K_crop_output, model_outputs['pose'])

            outputs[f'iteration={n+1}'] = {
                'TCO_input': TCO_input,
                'TCO_output': TCO_output,
                'K_crop_input': K_crop_input,
                'K_crop_output': K_crop_output,
                'K_input': K_input,
                'K_output': K_output,
                'model_outputs': model_outputs,
                'boxes_rend': boxes_rend,
                'boxes_crop': boxes_crop,
            }

            TCO_input = TCO_output

            if self.debug:
                self.tmp_debug.update(outputs[f'iteration={n+1}'])
                self.tmp_debug.update(
                    images=images,
                    images_crop=images_crop,
                    renders=renders,
                )
                path = DEBUG_DATA_DIR / f'debug_iter={n+1}.pth.tar'
                logger.info(f'Wrote debug data: {path}')
                torch.save(self.tmp_debug, path)

        return outputs
