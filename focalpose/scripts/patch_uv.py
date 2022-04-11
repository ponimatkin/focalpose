import os
import glob
from focalpose.config import LOCAL_DATA_DIR, MESHLAB_PTH


base_path = LOCAL_DATA_DIR / 'models_urdf' / 'pix3d'

meshlab_path = f'xvfb-run -a -s "-screen 0 800x600x24" {MESHLAB_PTH}'

for cls in base_path.iterdir():
    for inst in cls.iterdir():
        if len(glob.glob(inst.as_posix() + '/*.mtl')):
            has_mtl = True
        else:
            has_mtl = False
        for obj in inst.iterdir():
            if obj.name == obj.with_name(inst.name).with_suffix('.obj').name:
                print(f'Working on the mesh: {obj}')
                os.system(meshlab_path + ' -i ' + obj.as_posix() + ' -o ' + obj.as_posix() + ' -m vn vt wt -s focalpose/scripts/uv_map_filter.xml')
                if not has_mtl:
                    print(f'Generating empty texture for mesh: {obj}')
                    os.system(meshlab_path + ' -i ' + obj.as_posix() + ' -o ' + obj.as_posix() + ' -m vn vt wt -s focalpose/scripts/blank_texture_filter.xml')