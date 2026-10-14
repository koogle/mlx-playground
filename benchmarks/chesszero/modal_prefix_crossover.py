from pathlib import Path
import modal

import os
root=Path(os.environ.get("CHESSZERO_CROSSOVER_EXPORT_DIR", "/workspace/chesszero-ablation-crossover"))
script=Path(__file__).with_name("prefix_crossover.py")
image=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
       .apt_install('libgomp1').pip_install('mlx[cuda12]==0.32.3','numpy==2.2.6','psutil==7.0.0','tqdm==4.67.1','cachetools==7.2.1')
       .env({'MLX_CUDA_CONV_CACHE_SIZE':'1024','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TQDM_DISABLE':'1'})
       .add_local_dir(str(root/'legacy'),'/opt/legacy',copy=True)
       .add_local_dir(str(root/'current'),'/opt/current',copy=True)
       .add_local_file(str(script),'/opt/prefix.py',copy=True))
volume=modal.Volume.from_name('chesszero-convergence')
app=modal.App('chesszero-same-gpu-crossover')

@app.function(image=image,gpu='L40S',cpu=32,memory=(65536,393216),timeout=4200,scaledown_window=2,volumes={'/results':volume})
def crossover(run_id):
    import json,os,subprocess,sys,time,signal
    out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=False)
    cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(': \t')
    hardware={'cpu':cpu,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv,noheader'],text=True).strip()}
    (out/'hardware.json').write_text(json.dumps(hardware,indent=2));volume.commit()
    rows=[]
    for variant,mode in [('legacy','legacy'),('current','aligned'),('current','cached_no_tree'),('current','full_profiled'),('current','full_plain')]:
        phase=out/mode;phase.mkdir();start=time.monotonic();samples=[]
        with (phase/'run.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u','/opt/prefix.py',variant,mode,str(phase)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                try:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,clocks.current.sm,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                except subprocess.TimeoutExpired:gpu='telemetry timeout'
                samples.append({'seconds':time.monotonic()-start,'gpu':gpu})
                (phase/'samples.json').write_text(json.dumps(samples));volume.commit()
                print(mode,round(time.monotonic()-start),gpu,flush=True)
                if time.monotonic()-start>720:
                    os.killpg(process.pid,signal.SIGTERM);process.wait();break
                time.sleep(15)
        (phase/'job.json').write_text(json.dumps({'exit_code':process.returncode,'seconds':time.monotonic()-start}));volume.commit()
        if process.returncode:raise RuntimeError((phase/'run.log').read_text()[-3000:])
        rows.append(json.loads((phase/'result.json').read_text()));(out/'progress.json').write_text(json.dumps(rows,indent=2));volume.commit()
    (out/'result.json').write_text(json.dumps({'hardware':hardware,'phases':rows},indent=2));volume.commit();return {'hardware':hardware,'phases':rows}
