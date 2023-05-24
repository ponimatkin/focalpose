# Backbones
from focalpose.models.backbone import Backbone

# Pose models
from focalpose.models.pose import PosePredictor

from focalpose.utils.logging import get_logger
logger = get_logger(__name__)


def check_update_config(config):
    if not hasattr(config, 'init_method'):
        config.init_method = 'v0'
    return config


def create_model_pose(cfg, renderer, mesh_db):
    n_inputs = 6
    backbone_str = cfg.backbone_str
    pretrained = cfg.backbone_pretrained
    if 'resnet' in backbone_str:
        assert backbone_str in ['resnet18', 'resnet34', 'resnet50']
        backbone = Backbone(model=backbone_str, pretrained=pretrained)
    else:
        raise ValueError('Unknown backbone', backbone_str)

    pose_dim = cfg.n_pose_dims

    logger.info(f'Backbone: {backbone_str}')
    backbone.n_inputs = n_inputs
    render_size = (240, 320)
    model = PosePredictor(backbone=backbone,
                          renderer=renderer,
                          mesh_db=mesh_db,
                          render_size=render_size,
                          pose_dim=pose_dim)
    return model


def create_model_refiner(cfg, renderer, mesh_db):
    return create_model_pose(cfg, renderer, mesh_db)


def create_model_coarse(cfg, renderer, mesh_db):
    return create_model_pose(cfg, renderer, mesh_db)
