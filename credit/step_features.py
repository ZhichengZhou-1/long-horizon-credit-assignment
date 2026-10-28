"""Step embeddings for the RUDDER-style baseline: mean-pooled last hidden state of the policy model.

Each step (dump_steps.py) is embedded as one text: instruction, the action taken, then the page the
agent saw before it (truncated; instruction and action come first so they are never cut).
Run with the vllm env (torch + transformers) on a GPU, with no vLLM server holding the memory.
Output: OUT.npy (float16, one row per line of STEPS) and OUT.meta.json.

Usage: python step_features.py --steps steps.jsonl --out features [--model M] [--max-len 768]
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def step_text(s):
    return (f"Instruction: {s['instruction']}\nAction taken: {s['action']}\n"
            f"Page before the action: {s['obs']}")


@torch.no_grad()
def main(args):
    steps = [json.loads(l) for l in open(args.steps)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    enc = [tok(step_text(s), truncation=True, max_length=args.max_len)['input_ids'] for s in steps]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    feats = np.zeros((len(steps), model.config.hidden_size), dtype=np.float16)
    t0, n_tok, i, n_batch = time.time(), 0, 0, 0
    while i < len(order):
        # batches of roughly equal length, at most args.tokens tokens
        j = i
        while j < len(order) and (j - i + 1) * len(enc[order[j]]) <= args.tokens:
            j += 1
        j = max(j, i + 1)
        ids = [enc[k] for k in order[i:j]]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), tok.pad_token_id or 0, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for r, seq in enumerate(ids):
            x[r, :len(seq)] = torch.tensor(seq)
            m[r, :len(seq)] = 1
        x, m = x.cuda(), m.cuda()
        hid = model(input_ids=x, attention_mask=m).last_hidden_state.float()
        pooled = (hid * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        feats[order[i:j]] = pooled.cpu().numpy().astype(np.float16)
        n_tok += int(m.sum())
        i, n_batch = j, n_batch + 1
        if n_batch % 50 == 0 or i == len(order):
            print(f'{i}/{len(order)} steps, {n_tok / (time.time() - t0):.0f} tok/s', flush=True)
    np.save(args.out + '.npy', feats)
    json.dump({'model': args.model, 'max_len': args.max_len, 'steps': args.steps, 'n': len(steps),
               'dim': int(feats.shape[1]), 'tokens': n_tok, 'wall_s': round(time.time() - t0, 1)},
              open(args.out + '.meta.json', 'w'), indent=2)
    print(f'saved {feats.shape} to {args.out}.npy in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', default='langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop')
    p.add_argument('--max-len', type=int, default=768)
    p.add_argument('--tokens', type=int, default=32768, help='max tokens per batch')
    main(p.parse_args())
