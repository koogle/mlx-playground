import sys
sys.path.insert(0, '/workspace/chesszero-pipe-inference/chesszero')
import mlx.core as mx
import numpy as np
from model.pipe_inference import run_pipe_games
from config.model_config import ModelConfig
from model.training import Trainer
from model.self_play import play_single_game, create_batches
from mlx.utils import tree_flatten


def probe(client, kind, connection, workers):
    if kind == 'crash':
        import os
        os._exit(7)
    if kind == 'missing':
        return
    states=np.zeros((1,19,8,8), dtype=np.float32)
    policies,values=client(states)
    assert policies.shape==(1,4672)
    assert values.shape==(1,1)
    connection.send({'history':[], 'white_result':0})
    client.close()


if __name__=='__main__':
    mx.set_default_device(mx.cpu)
    config=ModelConfig(n_residual_blocks=0,n_filters=2,n_simulations=2,batch_size=64,n_games_per_iteration=4)
    trainer=Trainer(config,checkpoint_dir='/tmp/chesszero-pipe-smoke-checkpoints')
    rows,failures,stats=run_pipe_games(trainer.model,config,probe,[(x,) for x in ['normal','normal','crash','missing']],4)
    assert len(rows)==2 and failures==2
    assert stats['positions']==1,stats
    rows,failures,stats=run_pipe_games(trainer.model,config,play_single_game,[(config,i) for i in range(4)],4)
    assert len(rows)==4 and failures==0
    games=[(r['history'],r['white_result']) for r in rows]
    losses=trainer.train_on_batches(create_batches(games,64))
    assert all(np.isfinite(losses))
    assert int(trainer.optimizer.state['step'])==1
    assert all(bool(mx.all(mx.isfinite(p)).item()) for _,p in tree_flatten(trainer.model.parameters()))
    trainer._save_checkpoint(1)
    print('PASS',len(rows),'games',sum(len(g[0]) for g in games),'positions',stats,'losses',losses)
