"""Three fresh 512-game generations using the reviewed production model/search."""
import sys
from pathlib import Path
sys.path.insert(0, '/opt/chesszero/chesszero')
import gc
import json
import logging
import time
import numpy as np
from config.model_config import ModelConfig
from model.game_workers import run_game_workers
from model.inference import BatchedInference
from model.self_play import play_single_game, create_batches


def game_worker(client, config, game_id, generation, result_connection, max_workers):
    np.random.seed(9000 + generation * 10000 + game_id)
    play_single_game(client, config, game_id, result_connection, max_workers)


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.training import Trainer
    from model.network import ChessNet
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    source = Path('/results/learning-merged-20261013-205953/checkpoints')
    trainer = Trainer(ModelConfig(), checkpoint_dir=source, resume_epoch=4)
    trainer.checkpoint_dir = out / 'checkpoints'
    trainer.config.n_games_per_iteration = 512
    trainer.config.n_simulations = 32
    trainer.config.batch_size = 64
    trainer.config.inference_batch_size = 64
    trainer._accept_current_model()
    initial_step = int(trainer.optimizer.state['step'])
    rows = []
    for generation in range(3):
        trainer.model.eval()
        mx.eval(trainer.model.parameters())
        mx.clear_cache()
        started = time.monotonic()
        print('GENERATION', generation + 1, 'START', flush=True)
        with BatchedInference(trainer.model, trainer.config) as inference:
            results, failures = run_game_workers(
                game_worker,
                [(trainer.config, i, generation) for i in range(512)],
                512, clients=(inference.client(),),
            )
        seconds = time.monotonic() - started
        if failures or len(results) != 512:
            raise RuntimeError(f'Incomplete generation: {len(results)} games, {failures} failures')
        games = [(r['history'], r['white_result']) for r in results]
        row = dict(generation=generation + 1, games=len(games), failures=failures,
                   positions=sum(len(h) for h, _ in games), self_play_seconds=seconds,
                   games_per_minute=512 * 60 / seconds, inference=inference.statistics,
                   white_wins=sum(v == 1 for _, v in games),
                   black_wins=sum(v == -1 for _, v in games), draws=sum(v == 0 for _, v in games))
        (out / f'generation-{generation+1}.json').write_text(json.dumps(row, indent=2))
        print('SELF_PLAY_COMPLETE', json.dumps(row), flush=True)
        del results
        started = time.monotonic()
        def batches():
            for i, batch in enumerate(create_batches(games, trainer.config.batch_size)):
                if i % 100 == 0:
                    print('TRAINING', generation + 1, 'MICROBATCH', i, flush=True)
                yield batch
        row['losses'] = trainer.train_on_batches(batches())
        row['learning_seconds'] = time.monotonic() - started
        row['adam_step'] = int(trainer.optimizer.state['step'])
        assert row['adam_step'] == initial_step + generation + 1
        row['finite_parameters'] = all(bool(mx.all(mx.isfinite(v)).item())
                                     for _, v in tree_flatten(trainer.model.parameters()))
        if not row['finite_parameters']:
            raise RuntimeError('Nonfinite learned parameters')
        trainer._save_checkpoint(5 + generation)
        restored, state = ChessNet.load_checkpoint(trainer.checkpoint_dir, 5 + generation)
        assert int(state['optimizer_state']['step']) == row['adam_step']
        assert state['completed_epochs'] == 5 + generation
        assert all(bool(mx.array_equal(a, b).item()) for (_, a), (_, b) in zip(
            tree_flatten(trainer.model.parameters()), tree_flatten(restored.parameters())))
        row['checkpoint_roundtrip'] = True
        del restored, state, games
        gc.collect()
        mx.clear_cache()
        rows.append(row)
        (out / 'progress.json').write_text(json.dumps(rows, indent=2))
        print('GENERATION_COMPLETE', json.dumps(row), flush=True)
    trainer.model.eval()
    print('FINAL_EVALUATION_START 40 games vs starting model', flush=True)
    win_rate, wins, losses, draws = trainer.evaluate_against_best(n_games=40, max_workers=40)
    result = dict(source_revision=Path('/opt/chesszero/REVISION').read_text().strip(),
                  original_network=dict(n_residual_blocks=trainer.config.n_residual_blocks,
                                        n_filters=trainer.config.n_filters),
                  workers=512, n_simulations=32, microbatch_size=64, initial_adam_step=initial_step,
                  generations=rows,
                  evaluation=dict(games=40, wins=wins, losses=losses, draws=draws, win_rate=win_rate))
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('RUN_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
