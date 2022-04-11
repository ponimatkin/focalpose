import numpy as np
import os
from pathlib import Path
from focalpose.config import LOCAL_DATA_DIR
import yaml
import pybullet as pb
import pybullet_data
from focalpose.simulator.body import Body

def main():
    urdf_path = LOCAL_DATA_DIR / 'models_urdf' / 'pix3d'

    for obj in urdf_path.iterdir():
        scales = dict()
        cfg = dict()
        config_path = (LOCAL_DATA_DIR / 'configs' / os.path.basename(obj))
        config = config_path.with_suffix('.yaml')

        for instances in obj.iterdir():
            if not instances.is_dir():
                continue

            client_id = pb.connect(pb.DIRECT)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())

            model = Body.load((instances / instances.with_suffix('.urdf').name).as_posix(), client_id=client_id, scale=1.0)
            AABB = np.array(pb.getAABB(model._body_id, -1, client_id))
            kpts = np.loadtxt((instances / '3d_keypoints.txt').as_posix())
            kpts_min = np.array([min(kpts[:, 0]), min(kpts[:, 1]), min(kpts[:, 2])])
            kpts_max = np.array([max(kpts[:, 0]), max(kpts[:, 1]), max(kpts[:, 2])])
            scale = np.array([1, 1, 1]) #(AABB[1] - AABB[0])/(kpts_max - kpts_min)[0]

            scales[instances.name] = scale.tolist()

        scales = dict(sorted(scales.items()))
        cfg['scales'] = scales

        config.write_text(yaml.dump(cfg))


if __name__ == '__main__':
    main()
