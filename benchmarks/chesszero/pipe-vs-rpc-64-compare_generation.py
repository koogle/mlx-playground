"""Identical 512-game workload; branch-native inference and batching."""
import sys
from pathlib import Path
sys.path.insert(0, '/opt/chesszero/chesszero')
import json
import os
import time
import numpy as np
from config.model_config import ModelConfig
from model.self_play import play_single_game, create_batches
import model.self_play as self_play
from functools import partial
self_play.tqdm = partial(self_play.tqdm, disable=True)


def timed_game(model_inference, config, game_id, directory, result_connection, max_workers):
    if os.environ['CHESSZERO_VARIANT'] == 'unused_gpu_worker':
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.random.seed(9000 + game_id)
    assert 'mlx.core' not in sys.modules, 'CPU game worker imported MLX'
    np.random.seed(9000 + game_id)
    start = time.monotonic()
    play_single_game(model_inference, config, game_id, result_connection, max_workers)
    (Path(directory) / f'game-{game_id}.json').write_text(json.dumps({'seconds': time.monotonic()-start}))


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.training import Trainer
    from model.network import ChessNet
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    trainer = Trainer(ModelConfig(), checkpoint_dir='/results/learning-merged-20261013-205953/checkpoints', resume_epoch=4)
    trainer.checkpoint_dir = out / 'checkpoints'
    trainer.config.n_games_per_iteration = 512
    trainer.config.n_simulations = 64
    trainer.config.batch_size = 64
    variant = os.environ['CHESSZERO_VARIANT']
    trainer.config.inference_batch_size = 64
    if variant == 'rpc':
        trainer.config.inference_batch_size = 64
    trainer.model.eval()
    mx.eval(trainer.model.parameters())
    mx.clear_cache()
    initial_step = int(trainer.optimizer.state['step'])
    row = dict(variant=variant, workers=512, games_requested=512, n_simulations=64,
               seed_base=9000, initial_generation=4, initial_adam_step=initial_step,
               source_revision=Path('/opt/chesszero/REVISION').read_text().strip(),
               effective_config=vars(trainer.config), cuda_convolution_cache_size=os.environ['MLX_CUDA_CONV_CACHE_SIZE'])
    (out / 'configuration.json').write_text(json.dumps(row, indent=2))
    started = time.monotonic()
    results, failures, inference, error = [], 0, None, None
    try:
        if variant == 'rpc':
            from model.game_workers import run_game_workers
            from model.inference import BatchedInference
            inference = BatchedInference(trainer.model, trainer.config)
            with inference:
                results, failures = run_game_workers(
                    timed_game, [(trainer.config, i, str(out)) for i in range(512)],
                    512, clients=(inference.client(),))
        else:
            from model.pipe_inference import run_pipe_games
            results, failures, pipe_statistics = run_pipe_games(
                trainer.model, trainer.config, timed_game,
                [(trainer.config, i, str(out)) for i in range(512)], 512)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    row.update(games=len(results), failures=failures, self_play_seconds=time.monotonic()-started,
               positions=sum(len(r['history']) for r in results), generation_error=error)
    if variant == 'pipe':
        row['inference'] = pipe_statistics
    if inference is not None:
        row['inference'] = getattr(inference, 'statistics', None)
    (out / 'generation.json').write_text(json.dumps(row, indent=2))
    print('GENERATION_RESULT', json.dumps(row), flush=True)
    if error or failures or len(results) != 512:
        raise RuntimeError(f'Incomplete 512-game generation: {error}, {len(results)} completed, {failures} failed')
    games = [(r['history'], r['white_result']) for r in results]
    row['game_plies'] = [len(h) for h, _ in games]
    row['white_wins'] = sum(v == 1 for _, v in games)
    row['black_wins'] = sum(v == -1 for _, v in games)
    row['draws'] = sum(v == 0 for _, v in games)
    row['games_per_minute'] = len(games)*60/row['self_play_seconds']
    row['positions_per_second'] = row['positions']/row['self_play_seconds']
    started = time.monotonic()
    def batches():
        for i, batch in enumerate(create_batches(games, 64)):
            if i % 100 == 0:
                print('TRAINING_MICROBATCH', i, flush=True)
            yield batch
    row['losses'] = trainer.train_on_batches(batches())
    row['learning_seconds'] = time.monotonic()-started
    row['adam_step'] = int(trainer.optimizer.state['step'])
    assert row['adam_step'] == initial_step + 1
    row['finite_parameters'] = all(bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(trainer.model.parameters()))
    if not row['finite_parameters']:
        raise RuntimeError('Nonfinite parameters')
    trainer._save_checkpoint(5)
    restored, state = ChessNet.load_checkpoint(trainer.checkpoint_dir, 5)
    assert int(state['optimizer_state']['step']) == row['adam_step']
    assert all(bool(mx.array_equal(a, b).item()) for (_, a), (_, b) in zip(
        tree_flatten(trainer.model.parameters()), tree_flatten(restored.parameters())))
    row['checkpoint_roundtrip'] = True
    (out / 'result.json').write_text(json.dumps(row, indent=2))
    print('RESULT', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
