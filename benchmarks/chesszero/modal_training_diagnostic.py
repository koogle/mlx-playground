"""Check whether asynchronous Python stack dumping is safe during MLX autodiff."""

from pathlib import Path
import modal
from modal_batched_throughput import image, volume

image = image.add_local_python_source("modal_batched_throughput", copy=True)
app = modal.App("chesszero-training-diagnostic-full")


@app.function(
    image=image,
    gpu="L40S",
    cpu=32,
    memory=65536,
    timeout=900,
    volumes={"/results": volume},
)
def full_training(run_id):
    import subprocess
    import sys
    import time

    out = Path("/results") / run_id
    out.mkdir(parents=True, exist_ok=True)
    code = r"""
import sys,time,faulthandler,json
sys.path.insert(0,'/opt/chesszero/chesszero')
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from model.training import Trainer
from config.model_config import ModelConfig
from chess_engine.bitboard import BitBoard
mx.set_default_device(mx.gpu)
trainer=Trainer(ModelConfig(),checkpoint_dir='/results/learning-merged-20261013-205953/checkpoints',resume_epoch=4)
state=mx.array(np.repeat(BitBoard().state[None],64,axis=0))
policy=mx.ones((64,4672))/4672
value=mx.zeros(64)
t=time.monotonic()
def batches():
    for i in range(0,181645,64):
        count=min(64,181645-i)
        if i//64%100==0: print('microbatch',i//64,flush=True)
        yield state[:count],policy[:count],value[:count]
losses=trainer.train_on_batches(batches())
finite=all(bool(mx.all(mx.isfinite(x)).item()) for _,x in tree_flatten(trainer.model.parameters()))
print(json.dumps({'losses':losses,'step':int(trainer.optimizer.state['step']),'finite':finite,'seconds':time.monotonic()-t}),flush=True)
"""
    with (out / "run.log").open("w") as log:
        p = subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdout=log, stderr=subprocess.STDOUT
        )
        start = time.monotonic()
        while p.poll() is None:
            volume.commit()
            if time.monotonic() - start > 600:
                p.terminate()
                p.wait()
                break
            time.sleep(5)
    r = {"exit_code": p.returncode, "log": (out / "run.log").read_text()}
    (out / "result.json").write_text(__import__("json").dumps(r, indent=2))
    volume.commit()
    return r
