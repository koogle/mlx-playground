"""Fresh, bounded opening rollouts for same-host diagnostic timing; no learning."""
import sys
from pathlib import Path
VARIANT, MODE = sys.argv[1:3]
sys.path.insert(0, '/opt/'+VARIANT+'/chesszero')
import hashlib
import json
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from chess_engine.game import ChessGame
from chess_engine.bitboard import BitBoard
from model.mcts import MCTS, Node, GameSearchCache
from model.self_play import get_policy_distribution
from model.game_workers import run_game_workers
from model.inference import BatchedInference, _InferenceManager, InferenceClient
from config.model_config import ModelConfig


class NoPredictions(dict):
    def __setitem__(self,key,value):
        pass


def backend_factory(*args):
    if VARIANT == 'legacy':
        from model.inference import _InferenceBackend
    else:
        from model.inference_backend import _InferenceBackend
    if MODE == 'aligned':
        from queue import Empty
        def serve(self):
            import mlx.core as mx
            mx.set_default_device(self.device)
            while True:
                first=self.requests.get()
                if first is None:return
                batch=[first]
                while len(batch)<self.batch_size:
                    try:request=self.requests.get_nowait()
                    except Empty:break
                    if request is None:
                        self.requests.put(None);break
                    batch.append(request)
                pending={(snapshot,id(response)):(position,[response]) for snapshot,position,response in batch}
                try:
                    if self.error is not None:raise RuntimeError(self.error)
                    self._predict(pending)
                except Exception as error:
                    self.error=str(error)
                    for _,_,response in batch:
                        if not response.done():response.set_exception(RuntimeError(self.error))
        _InferenceBackend._serve=serve
    backend=_InferenceBackend(*args)
    if MODE == 'aligned':backend.predictions=NoPredictions()
    return backend


def worker(client,config,game_id,barrier,ready,sender,max_workers):
    assert 'mlx.core' not in sys.modules
    np.random.seed(9000+game_id)
    profile=Counter()
    if MODE != 'full_plain':
        def instrument(cls,name,key,timed=False):
            original=getattr(cls,name)
            def call(self,*args,**kwargs):
                profile[key]+=1
                if not timed:return original(self,*args,**kwargs)
                start=time.perf_counter();cpu=time.process_time()
                try:return original(self,*args,**kwargs)
                finally:
                    profile[key+'_seconds']+=time.perf_counter()-start
                    profile[key+'_cpu_seconds']+=time.process_time()-cpu
            setattr(cls,name,call)
        instrument(Node,'select_child','selection_steps')
        instrument(BitBoard,'copy','board_copies')
        instrument(MCTS,'_simulate','simulations',True)
        instrument(MCTS,'_expand_node','expansions',True)
        instrument(InferenceClient,'__call__','rpc_requests',True)
    if MODE in ('aligned','cached_no_tree'):
        MCTS.advance_root=lambda self,*args:setattr(self,'root_node',None)
    game=ChessGame();cache=GameSearchCache(client);search=MCTS(client,config,cache=cache)
    # Open RPC sockets as workers spawn, avoiding an artificial connection storm.
    client(np.asarray(game.board.state,dtype=np.float32)[None])
    profile.clear()
    with ready.get_lock():ready.value+=1
    barrier.wait(timeout=240)
    start=time.perf_counter();moves=[]
    try:
        for ply in range(32):
            if game.board.is_game_over():break
            temperature=config.temperature if ply<config.temp_decay_steps else config.temp_final
            move=search.get_move(game.board,temperature=temperature)
            if move is None:raise RuntimeError('Ongoing prefix has no move')
            get_policy_distribution(search.root_node,config.policy_output_dim)
            if not game.make_move(*move):raise RuntimeError('Illegal prefix move')
            if hasattr(search,'advance_root'):search.advance_root(move,game.board)
            moves.append(move)
        sender.send(dict(game_id=game_id,positions=len(moves),seconds=time.perf_counter()-start,
                         path_digest=hashlib.sha256(json.dumps(moves).encode()).hexdigest(),profile=dict(profile)))
    finally:
        cache.clear();client.close()


def main():
    import mlx.core as mx
    from model.training import Trainer
    mx.set_default_device(mx.gpu)
    out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
    trainer=Trainer(ModelConfig(),checkpoint_dir='/results/learning-merged-20261013-205953/checkpoints',resume_epoch=4)
    trainer.config.n_simulations=32;trainer.config.inference_batch_size=64
    trainer.model.eval();mx.eval(trainer.model.parameters());mx.clear_cache()
    _InferenceManager.register('backend',backend_factory,exposed=('predict','predict_reference','statistics','close'))
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(513);ready=ctx.Value('i',0)
    start=time.monotonic()
    with BatchedInference(trainer.model,trainer.config) as inference:
        with ThreadPoolExecutor(max_workers=1) as scheduler:
            result=scheduler.submit(run_game_workers,worker,[(trainer.config,i,barrier,ready) for i in range(512)],512,clients=(inference.client(),))
            deadline=time.monotonic()+240
            while ready.value<512:
                if result.done():result.result();raise RuntimeError('Workers ended before ready')
                if time.monotonic()>deadline:raise RuntimeError('Worker readiness timeout')
                time.sleep(0.1)
            before=inference.rpc.statistics()
            search_start=time.monotonic();barrier.wait(timeout=240)
            results,failures=result.result()
            search_seconds=time.monotonic()-search_start
    wall=time.monotonic()-start
    if failures or len(results)!=512:raise RuntimeError(f'Incomplete prefix test: {failures} failures')
    totals=dict(sum((Counter(r['profile']) for r in results),Counter()))
    statistics=dict(inference.statistics)
    for key in ('cache_hits','forwards','positions'):statistics[key]=statistics.get(key,0)-before.get(key,0)
    row=dict(variant=VARIANT,mode=MODE,workers=512,prefix_plies_limit=32,
             positions=sum(r['positions'] for r in results),wall_seconds=wall,search_seconds=search_seconds,
             max_worker_search_seconds=max(r['seconds'] for r in results),
             mean_worker_search_seconds=sum(r['seconds'] for r in results)/512,
             inference=statistics,profile=totals,path_digests=[r['path_digest'] for r in results],
             source_revision=Path('/opt/'+VARIANT+'/REVISION').read_text().strip())
    (out/'result.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
