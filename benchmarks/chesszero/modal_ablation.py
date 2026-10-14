from pathlib import Path
import os
import modal
from modal_batched_throughput import experiment, volume

root = Path(os.environ.get("CHESSZERO_ABLATION_EXPORT_DIR", "/workspace/chesszero-ablation"))
script = Path(__file__).with_name("ablation_generation.py")
base = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
        .apt_install('libgomp1')
        .pip_install('mlx[cuda12]==0.32.3','numpy==2.2.6','psutil==7.0.0','tqdm==4.67.1','cachetools==7.2.1')
        .env({'MLX_CUDA_CONV_CACHE_SIZE':'1024','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        .add_local_file(str(script), '/opt/throughput.py', copy=True)
        .add_local_python_source('modal_batched_throughput', copy=True))
candidate_image = base.add_local_dir(os.environ.get("CHESSZERO_ABLATION_CANDIDATE", "/workspace/chesszero-compare-512/candidate"), '/opt/chesszero', copy=True).env({'CHESSZERO_VARIANT':'candidate'})
baseline_image = base.add_local_dir(str(root / 'baseline27'), '/opt/chesszero', copy=True).env({'CHESSZERO_VARIANT':'main','CHESSZERO_ABLATION':'both'})
app = modal.App('chesszero-cache-tree-ablation')

@app.function(image=candidate_image,gpu='L40S',cpu=32,memory=(65536,393216),timeout=4500,
              scaledown_window=2,volumes={'/results':volume})
def ablation(run_id, mode):
    os.environ['CHESSZERO_ABLATION'] = mode
    return experiment(mode,run_id,512,limit_seconds=4200)

@app.function(image=baseline_image,gpu='L40S',cpu=32,memory=(65536,393216),timeout=3900,
              scaledown_window=2,volumes={'/results':volume})
def baseline27(run_id):
    return experiment('baseline27',run_id,512,limit_seconds=3600)
