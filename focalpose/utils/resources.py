import os
import psutil
import shutil
import subprocess
from shutil import which
import xml.etree.ElementTree as ET

def is_egl_available():
    return is_gpu_available and 'EGL_VISIBLE_DEVICES' in os.environ


def is_gpu_available():
    return which('nvidia-smi') is not None


def is_slurm_available():
    return which('sinfo') is not None


def get_total_memory():
    current_process = psutil.Process(os.getpid())
    mem = current_process.memory_info().rss
    for child in current_process.children(recursive=True):
        mem += child.memory_info().rss
    return mem / 1e9

def assign_gpu():
    if shutil.which('squeue'):
        device_ids = os.environ['CUDA_VISIBLE_DEVICES']
        device_ids = device_ids.split(',')
        slurm_localid = int(os.environ['SLURM_LOCALID'])
        assert slurm_localid < len(device_ids)
        cuda_id = int(device_ids[slurm_localid])
    elif shutil.which('qstat'):
        cuda_id = int(os.environ['MPI_LOCALRANKID'])

    cuda_id = 0
    out = subprocess.check_output(['nvidia-smi', '-q', '--xml-format'])
    tree = ET.fromstring(out)
    gpus = tree.findall('gpu')
    gpu = gpus[cuda_id]
    dev_id = gpu.find('minor_number').text

    os.environ['EGL_VISIBLE_DEVICES'] = str(dev_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)