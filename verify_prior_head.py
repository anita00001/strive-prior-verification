"""Bound *original pretrained STRIVE conditional-prior output head* using auto_LiRPA.

Claim is conditional on its 128-D intermediate input being in the declared box.
Does NOT verify upstream map CNN/GNN, downstream decoder, or physical safety.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def load_original_prior_head(ckpt_file):
    # The original STRIVE MLP([128,128,128,64]) has this exact layer order.
    layers = nn.Sequential(
        nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 64),
    )
    # WARNING: Load only checkpoints you trust; old checkpoints may contain unsafe pickles.
    raw = torch.load(ckpt_file, map_location='cpu')
    state = raw.get('model', raw.get('state_dict', raw))
    prefix = 'prior_net.mlp_out.net.'
    sub = {key[len(prefix):]: tensor for key, tensor in state.items()
           if key.startswith(prefix)}
    if not sub:
        raise ValueError(f'No {prefix}* keys found: wrong STRIVE checkpoint?')
    layers.load_state_dict(sub, strict=True)
    return layers.eval()


def verify(checkpoint, capture, epsilon=1e-3, delta=0.05,
           method='backward', output_file='prior_result.json'):
    if not (epsilon > 0 and delta > 0):
        raise ValueError('epsilon and delta must be positive')
    head = load_original_prior_head(checkpoint)
    record = torch.load(capture, map_location='cpu')
    h0 = record['hidden'].reshape(1, 128).float()
    recorded = record['original_output'].reshape(1, 64).float()
    with torch.no_grad():
        reference = head(h0)
    # Fail closed on mismatch rather than certifying a different checkpoint.
    if not torch.allclose(reference, recorded, atol=1e-4, rtol=1e-4):
        raise RuntimeError('Original forward-hook output differs from reloaded pretrained head!')
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    perturbed = BoundedTensor(h0, PerturbationLpNorm(norm=float('inf'), eps=epsilon))
    bounded_head = BoundedModule(head, h0, device='cpu')
    if method == 'CROWN-Optimized':
        bounded_head.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    lower, upper = bounded_head.compute_bounds(x=(perturbed,), method=method)
    lower, upper = lower.detach(), upper.detach()
    # First 32 outputs are mu; final 32 outputs are log-variance before exp.
    # We check mean component 0 and log-variance component 0 as a smoke test.
    selected = (('mu_0', 0), ('logvar_0', 32))
    outcomes = {}
    for name, idx in selected:
        base, lo, hi = (float(reference[0, idx]), float(lower[0, idx]),
                        float(upper[0, idx]))
        success = lo >= base - delta and hi <= base + delta
        outcomes[name] = {'nominal': base, 'certified_lower': lo,
                          'certified_upper': hi, 'allowed_deviation': delta,
                          'verdict': 'CERTIFIED' if success else 'NOT_PROVED'}
        print(f'{name:10s} nominal={base:+.6f} certified_interval=[{lo:+.6f}, {hi:+.6f}]'
              f'  verdict={outcomes[name]["verdict"]}')
    sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    result = {
        'component': 'Original pretrained prior_net.mlp_out (MLP incl. LayerNorm)',
        'input_space': 'Independent Linf perturbation of captured 128-D hidden activation',
        'checkpoint_sha256': sha256,
        'epsilon': epsilon, 'delta': delta, 'method': method,
        'scope_warning': 'No claim about complete prior GNN, map encoder, decoder, or vehicle safety',
        'outputs': outcomes,
    }
    Path(output_file).write_text(json.dumps(result, indent=2))
    print('Wrote', output_file)
    return all(o['verdict'] == 'CERTIFIED' for o in outcomes.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--capture', default='prior_head_capture.pt')
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--method', choices=['IBP', 'backward', 'CROWN-Optimized'], default='backward')
    parser.add_argument('--out', default='prior_result.json')
    opts = parser.parse_args()
    try:
        ok = verify(opts.ckpt, opts.capture, opts.epsilon, opts.delta, opts.method, opts.out)
        # NOT_PROVED is not a counterexample; bound could simply be too loose.
        raise SystemExit(0 if ok else 2)
    except Exception as exc:
        print('NO VERIFICATION RESULT:', type(exc).__name__, str(exc))
        raise SystemExit(3)
