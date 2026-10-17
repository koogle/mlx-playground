"""Measure how reproducible self-play policy targets are. Run: python utils/target_noise_diagnostic.py"""
import sys, logging
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, mlx.core as mx
from config.model_config import ModelConfig
from model.network import ChessNet
from model.mcts import MCTS
from model.self_play import get_policy_distribution
from chess_engine.bitboard import BitBoard
from utils.random_player import RandomPlayer
logging.basicConfig(level=logging.WARNING)
rng = np.random.default_rng(0)
# 12 positions from random play at plies 4..40
rp = RandomPlayer(); boards=[]
for i in range(12):
    b=BitBoard(); n=int(rng.integers(4,40))
    for _ in range(n):
        if b.is_game_over(): break
        b.make_move(*rp.select_move(b))
    if not b.is_game_over(): boards.append(b)
cfg0 = ModelConfig(n_residual_blocks=1, n_filters=16)
model = ChessNet(cfg0); model.eval()
def search(board, sims, c_puct, noise, seed):
    np.random.seed(seed)
    cfg = ModelConfig(n_residual_blocks=1, n_filters=16, n_simulations=sims, c_puct=c_puct)
    m = MCTS(model, cfg, exploration_noise=noise)
    m.get_move(board.copy(), temperature=1.0)
    p = np.array(get_policy_distribution(m.root_node, cfg.policy_output_dim))
    return p[p>0] if False else p
def js(p,q):
    m=(p+q)/2
    def kl(a,b):
        mask=a>0; return float(np.sum(a[mask]*np.log(a[mask]/b[mask])))
    return 0.5*kl(p,m)+0.5*kl(q,m)
for sims, c, noise in [(64,1,True),(64,1,False),(256,1,True),(64,3,True),(256,3,True),(800,3,False)]:
    js_rr=[]; js_ru=[]; agree=0; ent=[]
    for b in boards:
        p1=search(b,sims,c,noise,1); p2=search(b,sims,c,noise,2)
        legal=p1>0; u=np.zeros_like(p1); u[legal]=1/legal.sum()
        js_rr.append(js(p1,p2)); js_ru.append((js(p1,u)+js(p2,u))/2)
        agree += int(np.argmax(p1)==np.argmax(p2))
        ent.append(-(p1[legal]*np.log(p1[legal])).sum()/np.log(legal.sum()))
    print(f"sims={sims:4d} c_puct={c} noise={noise!s:5}: JS(run1,run2)={np.mean(js_rr):.3f}  JS(run,uniform)={np.mean(js_ru):.3f}  argmax agree={agree}/{len(boards)}  rel.entropy={np.mean(ent):.2f}", flush=True)
