from pathlib import Path
import modal
from modal_batched_throughput import experiment, volume

root = Path(__file__).parent
base = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
        .apt_install('libgomp1')
        .pip_install('mlx[cuda12]==0.32.3','numpy==2.2.6','psutil==7.0.0','tqdm==4.67.1','cachetools==7.2.1')
        .env({'MLX_CUDA_CONV_CACHE_SIZE':'1024', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
        .add_local_file(str(root / 'compare_generation.py'), '/opt/throughput.py', copy=True)
        .add_local_python_source('modal_batched_throughput', copy=True))
pipe_image = base.add_local_dir(str(root / 'pipe'), '/opt/chesszero', copy=True).env({'CHESSZERO_VARIANT':'pipe'})
rpc_image = base.add_local_dir(str(Path('/workspace/chesszero-compare-512/candidate')), '/opt/chesszero', copy=True).env({'CHESSZERO_VARIANT':'rpc'})
app = modal.App('chesszero-pipe-vs-rpc-512-64sim')

@app.function(image=pipe_image, gpu='L40S', cpu=32, memory=(65536,393216), timeout=7500,
              scaledown_window=2, volumes={'/results':volume})
def pipe512(run_id):
    return experiment('pipe', run_id, 512, limit_seconds=7200)

@app.function(image=rpc_image, gpu='L40S', cpu=32, memory=(65536,393216), timeout=7500,
              scaledown_window=2, volumes={'/results':volume})
def rpc512(run_id):
    return experiment('rpc', run_id, 512, limit_seconds=7200)
