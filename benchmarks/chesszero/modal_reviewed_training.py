from pathlib import Path
import modal
from modal_batched_throughput import experiment, volume

image = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
         .apt_install('libgomp1')
         .pip_install('mlx[cuda12]==0.32.3', 'numpy==2.2.6', 'psutil==7.0.0', 'tqdm==4.67.1', 'cachetools==7.2.1')
         .add_local_dir('/workspace/chesszero-training-current-export/candidate', '/opt/chesszero', copy=True)
         .add_local_file(str(Path(__file__).with_name('reviewed_training.py')), '/opt/throughput.py', copy=True)
         .add_local_python_source('modal_batched_throughput', copy=True))
app = modal.App('chesszero-reviewed-training-512')

@app.function(image=image, gpu='L40S', cpu=32, memory=(65536,393216), timeout=14700,
              scaledown_window=2, volumes={'/results': volume})
def train512(run_id):
    return experiment('L40S', run_id, 512, limit_seconds=14400)
