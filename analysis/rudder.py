"""RUDDER-style baseline: learn to predict the final score from the step sequence, then credit each
step with the change in prediction it causes (return decomposition, Arjona-Medina et al. 2019).

Model: step embeddings (step_features.py) -> linear -> LSTM -> predicted final score g_t after each
step. Trained with MSE to the episode's final score at every step (the last step weighted most).
Contribution of step t: c_t = g_t - g_{t-1}, with g_{-1} = mean score of the training set.
The step blamed for a failure is the one with the most negative c_t.
Cross-fitted: goals are split into K folds and every episode is scored by a model that never saw
its goal, so the predictor cannot memorise the trajectories it is asked to explain.

Usage (vllm env python, CPU is enough):
  python rudder.py --data STEPS1.jsonl:FEATS1.npy [--data ...] --out rudder.jsonl
Output: one line per step of every episode: {src, idx, k, t, g, contrib, reward}.
"""
import argparse
import collections
import json
import random
import statistics

import numpy as np
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        y, _ = self.lstm(self.proj(x))
        return self.head(y).squeeze(-1)


def load(data):
    eps = collections.OrderedDict()
    for spec in data:
        steps_path, feat_path = spec.split(':')
        feats = np.load(feat_path).astype(np.float32)
        for i, line in enumerate(open(steps_path)):
            s = json.loads(line)
            key = (s['src'], s['idx'], s['k'])
            e = eps.setdefault(key, {'idx': s['idx'], 'reward': s['reward'], 'x': []})
            e['x'].append(feats[i])
    for e in eps.values():
        e['x'] = torch.tensor(np.stack(e['x']))
    return eps


def batches(keys, eps, bs, rng):
    keys = list(keys)
    rng.shuffle(keys)
    for i in range(0, len(keys), bs):
        chunk = keys[i:i + bs]
        L = max(len(eps[k]['x']) for k in chunk)
        x = torch.zeros(len(chunk), L, eps[chunk[0]]['x'].shape[1])
        mask = torch.zeros(len(chunk), L)
        y = torch.tensor([eps[k]['reward'] for k in chunk], dtype=torch.float32)
        for j, k in enumerate(chunk):
            n = len(eps[k]['x'])
            x[j, :n] = eps[k]['x']
            mask[j, :n] = 1
            mask[j, n - 1] = 1 + args.last_weight  # the final prediction is what RUDDER needs right
        yield x, mask, y


def train(keys, eps, rng):
    torch.manual_seed(rng.randrange(1 << 30))
    dim = next(iter(eps.values()))['x'].shape[1]
    model = Predictor(dim, args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    for _ in range(args.epochs):
        model.train()
        for x, mask, y in batches(keys, eps, args.bs, rng):
            g = model(x)
            loss = (mask * (g - y[:, None]) ** 2).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def main():
    rng = random.Random(args.seed)
    eps = load(args.data)
    goals = sorted({e['idx'] for e in eps.values()})
    rng.shuffle(goals)
    fold = {g: i % args.folds for i, g in enumerate(goals)}
    out, mse_last, mse_base = [], [], []
    for f in range(args.folds):
        tr = [k for k, e in eps.items() if fold[e['idx']] != f]
        te = [k for k, e in eps.items() if fold[e['idx']] == f]
        base = statistics.mean(eps[k]['reward'] for k in tr)
        model = train(tr, eps, rng)
        with torch.no_grad():
            for k in te:
                g = model(eps[k]['x'][None])[0].tolist()
                prev = [base] + g[:-1]
                mse_last.append((g[-1] - eps[k]['reward']) ** 2)
                mse_base.append((base - eps[k]['reward']) ** 2)
                for t, (gt, gp) in enumerate(zip(g, prev)):
                    out.append({'src': k[0], 'idx': k[1], 'k': k[2], 't': t, 'g': round(gt, 4),
                                'contrib': round(gt - gp, 4), 'reward': eps[k]['reward']})
        print(f'fold {f}: train {len(tr)} episodes, held out {len(te)}', flush=True)
    with open(args.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(f'{len(eps)} episodes, {len(goals)} goals, {args.folds} folds; held-out MSE of the final prediction '
          f'{statistics.mean(mse_last):.4f} vs predicting the training mean {statistics.mean(mse_base):.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', action='append', required=True, help='STEPS.jsonl:FEATURES.npy (repeatable)')
    p.add_argument('--out', required=True)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--last-weight', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main()
