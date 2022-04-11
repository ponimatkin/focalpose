from focalpose.config import LOCAL_DATA_DIR
from pathlib import Path
from shutil import copy2
from distutils.dir_util import copy_tree
import xml.etree.ElementTree as ET
from xml.dom import minidom


def obj_to_urdf(obj_path, urdf_path):
    obj_path = Path(obj_path)
    urdf_path = Path(urdf_path)
    assert urdf_path.parent == obj_path.parent

    geometry = ET.Element('geometry')
    mesh = ET.SubElement(geometry, 'mesh')
    mesh.set('filename', obj_path.name)
    mesh.set('scale', '1.0 1.0 1.0')

    material = ET.Element('material')
    material.set('name', 'mat_part0')
    color = ET.SubElement(material, 'color')
    color.set('rgba', '1.0 1.0 1.0 1.0')

    inertial = ET.Element('inertial')
    origin = ET.SubElement(inertial, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0.0 0.0 0.0')

    mass = ET.SubElement(inertial, 'mass')
    mass.set('value', '0.1')

    inertia = ET.SubElement(inertial, 'inertia')
    inertia.set('ixx', '1')
    inertia.set('ixy', '0')
    inertia.set('ixz', '0')
    inertia.set('iyy', '1')
    inertia.set('iyz', '0')
    inertia.set('izz', '1')

    robot = ET.Element('robot')
    robot.set('name', obj_path.with_suffix('').name)

    link = ET.SubElement(robot, 'link')
    link.set('name', 'base_link')

    visual = ET.SubElement(link, 'visual')
    origin = ET.SubElement(visual, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0.0 0.0 0.0')
    visual.append(geometry)
    visual.append(material)

    collision = ET.SubElement(link, 'collision')
    origin = ET.SubElement(collision, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0.0 0.0 0.0')
    collision.append(geometry)

    link.append(inertial)

    xmlstr = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="   ")
    Path(urdf_path).write_text(xmlstr)  # Write xml file


if __name__ == '__main__':
    urdf_path = LOCAL_DATA_DIR  / 'models_urdf' / 'StanfordCars3D'
    urdf_path.mkdir(exist_ok=True, parents=True)
    for cls in (LOCAL_DATA_DIR / 'StanfordCars' / 'models').iterdir():
        cls_path = urdf_path / cls.name
        cls_path.mkdir(exist_ok=True, parents=True)

        for obj in cls.iterdir():
            if obj.suffix == '.obj':
                copy2(obj.as_posix(), obj.with_name(f'{cls.name}').with_suffix('.obj'))
                obj_to_urdf(obj.with_name(f'{cls.name}').with_suffix('.obj'),
                            obj.with_name(f'{cls.name}').with_suffix('.urdf'))
        copy_tree(cls.as_posix(), cls_path.as_posix())

    urdf_path = LOCAL_DATA_DIR / 'models_urdf' / 'CompCars3D'
    urdf_path.mkdir(exist_ok=True, parents=True)
    for cls in (LOCAL_DATA_DIR / 'CompCars' / 'models').iterdir():
        cls_path = urdf_path / cls.name
        cls_path.mkdir(exist_ok=True, parents=True)

        for obj in cls.iterdir():
            if obj.suffix == '.obj':
                copy2(obj.as_posix(), obj.with_name(f'{cls.name}').with_suffix('.obj'))
                obj_to_urdf(obj.with_name(f'{cls.name}').with_suffix('.obj'),
                            obj.with_name(f'{cls.name}').with_suffix('.urdf'))
        copy_tree(cls.as_posix(), cls_path.as_posix())
