import importlib
import os
import sys

from .args import args


dir_path = os.path.dirname(os.path.realpath(__file__))
ezvtb_path = os.path.join(dir_path, "..", "ezvtuber-rt")
ezvtb_main_path = os.path.join(dir_path, "..", 'ezvtuber-rt-main')

project_path = ''
if os.path.exists(ezvtb_path):
    project_path = ezvtb_path
else:
    project_path = ezvtb_main_path

if project_path not in sys.path:
    sys.path.append(project_path)

# Initialize model data path to point to data/models directory.  The runtime
# package currently imports its TensorRT bootstrap eagerly, which initializes
# PyCUDA even when the application is about to select CoreORT.  In DirectML
# mode, temporarily mark that internal module unavailable so the package takes
# its existing "CoreTRT disabled" branch without loading CUDA/TensorRT.
_trt_bootstrap_module = 'ezvtb_rt.trt_utils'
_block_trt_bootstrap = (
    not args.use_tensorrt and _trt_bootstrap_module not in sys.modules
)
if _block_trt_bootstrap:
    sys.modules[_trt_bootstrap_module] = None
try:
    ezvtb_rt = importlib.import_module('ezvtb_rt')
finally:
    if _block_trt_bootstrap:
        sys.modules.pop(_trt_bootstrap_module, None)

models_path = os.path.join(dir_path, '..', 'data', 'models')
ezvtb_rt.init_model_path(models_path)

def get_core(
        use_tensorrt:bool = True, 
        #THA3 model setting
        model_version:str = 'v3', #'v3' or 'v4' or 'v4_student'
        model_name:str = '',
        model_seperable:bool = True,
        model_half:bool = True, #If using directml+half, there is small numerical error on Nvidia, and huge numerical error on AMD
        model_cache_size:float = 1.0, #unit of GigaBytes, only works for tensorrt
        model_use_eyebrow:bool = True,
        #RIFE interpolation setting
        use_interpolation:bool = True,
        interpolation_scale:int = 2,
        interpolation_half:bool = True, #If using directml+half, there is small numerical error on Nvidia, and huge numerical error on AMD
        #Cacher setting
        cacher_ram_size:float = 2.0,#unit of GigaBytes
        #SR setting
        use_sr:bool = False,
        sr_x4:bool = True,
        sr_half:bool = True,
        sr_a4k:bool = False,
        ):
    if use_tensorrt:
        try:
            from ezvtb_rt.core_trt import CoreTRT as Core
        except:
            print("TensorRT is not available, fallback to ONNX Runtime.")
            args.use_tensorrt = False
            from ezvtb_rt.core_ort import CoreORT as Core
    else:
        from ezvtb_rt.core_ort import CoreORT as Core
    core = Core(
        tha_model_version=model_version,
        tha_model_seperable=model_seperable,
        tha_model_fp16=model_half,
        tha_model_name=model_name,

        rife_model_enable=use_interpolation,
        rife_model_scale=interpolation_scale,
        rife_model_fp16=interpolation_half,

        sr_model_enable=use_sr,
        sr_model_scale=4 if sr_x4 else 2,
        sr_model_fp16=sr_half,
        sr_a4k=sr_a4k,

        vram_cache_size=model_cache_size,
        cache_max_giga=cacher_ram_size,
        use_eyebrow=model_use_eyebrow,
    )
    return core
