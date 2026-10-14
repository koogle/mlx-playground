"""Identical 512-game workload; branch-native inference and batching."""
import sys
from pathlib import Path
sys.path.insert(0, '/opt/chesszero/chesszero')
import json
import os
import time
from collections import Counter
import numpy as np
from config.model_config import ModelConfig
from model.self_play import play_single_game, create_batches
from model.game_workers import run_game_workers


# These hooks live only in the experiment harness, not the PR's production code.
class NoPredictions(dict):
    def __setitem__(self, key, value):
        pass


def make_backend(*args):
    from model.inference_backend import _InferenceBackend
    mode = os.environ['CHESSZERO_ABLATION']
    if mode == 'both_raw_batch':
        from queue import Empty
        def serve(self):
            import mlx.core as mx
            mx.set_default_device(self.device)
            while True:
                request = self.requests.get()
                if request is None:
                    return
                batch = [request]
                while len(batch) < self.batch_size:
                    try:
                        request = self.requests.get_nowait()
                    except Empty:
                        break
                    if request is None:
                        self.requests.put(None)
                        break
                    batch.append(request)
                pending = {}
                try:
                    if self.error is not None:
                        raise RuntimeError(self.error)
                    for snapshot, position, response in batch:
                        key = (snapshot, position.shape, position.tobytes())
                        if key in self.predictions:
                            self.cache_hits += 1
                            response.set_result(self.predictions[key])
                        elif key in pending:
                            self.cache_hits += 1
                            pending[key][1].append(response)
                        else:
                            pending[key] = (position, [response])
                    if pending:
                        self._predict(pending)
                except Exception as error:
                    self.error = str(error)
                    for _, _, response in batch:
                        if not response.done():
                            response.set_exception(RuntimeError(self.error))
        _InferenceBackend._serve = serve
    backend = _InferenceBackend(*args)
    if mode in ('neither', 'tree_only'):
        backend.predictions = NoPredictions()
    return backend


class RecordingSender:
    def __init__(self, connection, profile, directory, game_id):
        self.connection, self.profile = connection, profile
        self.directory, self.game_id = directory, game_id

    def send(self, result):
        if result is not None:
            result['profile'] = dict(self.profile)
            (Path(self.directory) / f'game-{self.game_id}.json').write_text(json.dumps(result['profile']))
        self.connection.send(result)


def timed_game(model_inference, config, game_id, directory, result_connection, max_workers):
    if os.environ['CHESSZERO_VARIANT'] == 'main':
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.random.seed(9000 + game_id)
    np.random.seed(9000 + game_id)
    from model.mcts import MCTS, Node, GameSearchCache
    from chess_engine.bitboard import BitBoard
    profile = Counter()
    mode = os.environ['CHESSZERO_ABLATION']
    if mode in ('neither', 'tree_only'):
        original_init = GameSearchCache.__init__
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.evaluations = NoPredictions()
        GameSearchCache.__init__ = init
    if mode in ('neither', 'cache_only'):
        def advance(self, *args):
            self.root_node = None
        MCTS.advance_root = advance
    def count_calls(cls, name, counter, timed=False):
        original = getattr(cls, name)
        def wrapped(self, *args, **kwargs):
            profile[counter] += 1
            if not timed:
                return original(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                profile[counter + '_seconds'] += time.perf_counter() - start
        setattr(cls, name, wrapped)
    count_calls(Node, 'select_child', 'selection_steps')
    count_calls(BitBoard, 'copy', 'board_copies')
    count_calls(MCTS, '_simulate', 'simulations', True)
    count_calls(MCTS, '_expand_node', 'expansions', True)
    count_calls(MCTS, '_evaluate_board', 'evaluations', True)
    if os.environ['CHESSZERO_VARIANT'] == 'candidate':
        from model.inference import InferenceClient
        count_calls(InferenceClient, '__call__', 'rpc_requests', True)
    start = time.monotonic()
    play_single_game(model_inference, config, game_id,
                     RecordingSender(result_connection, profile, directory, game_id), max_workers)
    profile['game_seconds'] = time.monotonic() - start
    (Path(directory) / f'game-{game_id}.json').write_text(json.dumps(dict(profile)))


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.training import Trainer
    from model.network import ChessNet
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    trainer = Trainer(ModelConfig(), checkpoint_dir='/results/learning-merged-20261013-205953/checkpoints', resume_epoch=4)
    trainer.checkpoint_dir = out / 'checkpoints'
    n_games = 16 if os.environ['CHESSZERO_VARIANT'] == 'main' else 512
    trainer.config.n_games_per_iteration = n_games
    trainer.config.n_simulations = 32
    trainer.config.batch_size = 64
    variant = os.environ['CHESSZERO_VARIANT']
    if variant == 'candidate':
        trainer.config.inference_batch_size = 64
    trainer.model.eval()
    mx.eval(trainer.model.parameters())
    mx.clear_cache()
    initial_step = int(trainer.optimizer.state['step'])
    row = dict(variant=variant, workers=512, games_requested=n_games, n_simulations=32,
               seed_base=9000, initial_generation=4, initial_adam_step=initial_step,
               source_revision=Path('/opt/chesszero/REVISION').read_text().strip(),
               effective_config=vars(trainer.config), cuda_convolution_cache_size=os.environ['MLX_CUDA_CONV_CACHE_SIZE'])
    (out / 'configuration.json').write_text(json.dumps(row, indent=2))
    row['ablation'] = os.environ['CHESSZERO_ABLATION']
    started = time.monotonic()
    results, failures, inference, error = [], 0, None, None
    try:
        if variant == 'candidate':
            from model.inference import BatchedInference, _InferenceManager
            _InferenceManager.register('backend', make_backend, exposed=('predict','predict_reference','statistics','close'))
            inference = BatchedInference(trainer.model, trainer.config)
            with inference:
                results, failures = run_game_workers(
                    timed_game, [(trainer.config, i, str(out)) for i in range(n_games)],
                    512, clients=(inference.client(),))
        else:
            results, failures = run_game_workers(
                timed_game, [(trainer.model, trainer.config, i, str(out)) for i in range(n_games)], 512)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    row.update(games=len(results), failures=failures, self_play_seconds=time.monotonic()-started,
               positions=sum(len(r['history']) for r in results), generation_error=error)
    if inference is not None:
        row['inference'] = getattr(inference, 'statistics', None)
    (out / 'generation.json').write_text(json.dumps(row, indent=2))
    print('GENERATION_RESULT', json.dumps(row), flush=True)
    if error or failures or len(results) != n_games:
        raise RuntimeError(f'Incomplete 512-game generation: {error}, {len(results)} completed, {failures} failed')
    games = [(r['history'], r['white_result']) for r in results]
    row['profile_totals'] = dict(sum((Counter(r['profile']) for r in results), Counter()))
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
