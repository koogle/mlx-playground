"""Isolate CUDA training-data preparation at the failed generation size."""

from pathlib import Path
import modal
from modal_batched_throughput import image, volume

image = image.add_local_python_source("modal_batched_throughput", copy=True)
app = modal.App("chesszero-batch-diagnostic")


@app.function(
    image=image,
    gpu="L40S",
    cpu=32,
    memory=(65536, 196608),
    timeout=900,
    volumes={"/results": volume},
)
def isolate_bulk(run_id, bulk=True):
    import subprocess
    import sys
    import time

    root = Path("/results") / run_id
    root.mkdir(parents=True, exist_ok=True)
    script = r"""
import sys,time,faulthandler
sys.path.insert(0,'/opt/chesszero/chesszero')
import numpy as np
import mlx.core as mx
from model.self_play import create_batches
faulthandler.dump_traceback_later(60,repeat=True)
mx.set_default_device(mx.gpu)
state=np.zeros((19,8,8),dtype=np.float32)
policy=np.zeros(4672,dtype=np.float32)
state[12]=1
policy[0]=1
# Independent arrays reproduce worker results rather than aliasing one fixture.
t=time.monotonic()
history=[(state.copy(),policy.copy(),0.) for _ in range(92318)]
print('fixture prepared',time.monotonic()-t,flush=True)
t=time.monotonic()
if BULK:
    stacked=np.stack([x[1] for x in history])
    print('NumPy policy stack completed',time.monotonic()-t,'bytes',stacked.nbytes,flush=True)
    policies=mx.array(stacked)
    print('bulk CUDA policy upload returned',time.monotonic()-t,flush=True)
    mx.eval(policies)
    print('bulk CUDA policy upload evaluated',time.monotonic()-t,flush=True)
    sys.exit(0)
batches=create_batches([(history,1.)],64)
batch=next(batches)
print('first batch returned',time.monotonic()-t,flush=True)
mx.eval(batch)
print('first batch evaluated',time.monotonic()-t,flush=True)
count=len(batch[0])
for batch in batches:
    mx.eval(batch)
    count+=len(batch[0])
print('all positions uploaded',count,'seconds',time.monotonic()-t,flush=True)
faulthandler.cancel_dump_traceback_later()
"""
    script = script.replace("BULK", str(bulk))
    with (root / "diagnostic.log").open("w") as log:
        p = subprocess.Popen(
            [sys.executable, "-u", "-c", script], stdout=log, stderr=subprocess.STDOUT
        )
        t = time.monotonic()
        while p.poll() is None:
            volume.commit()
            if time.monotonic() - t > 150:
                p.terminate()
                p.wait()
                break
            time.sleep(10)
    volume.commit()
    return {
        "exit_code": p.returncode,
        "seconds": time.monotonic() - t,
        "log": (root / "diagnostic.log").read_text(),
    }
