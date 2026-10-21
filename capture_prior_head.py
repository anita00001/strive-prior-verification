"""Install a one-shot PyTorch forward hook on STRIVE's conditional-prior output head.

Run ONLY with a pretrained model inside the original STRIVE test_traffic.py.
"""
from pathlib import Path
import torch


def attach_capture(model, filename='prior_head_capture.pt'):
    target = model.prior_net.mlp_out
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    handle = [None]
    captured = [False]

    def on_first_forward(module, inputs, output):
        if captured[0]:
            return
        h = inputs[0].detach().cpu()
        y = output.detach().cpu()
        if h.ndim != 2 or y.ndim != 2:
            raise ValueError(f'Unexpected shapes: {tuple(h.shape)}, {tuple(y.shape)}')
        # First *agent* in the first call, not every agent in a batch.
        if h.shape[1] != 128 or y.shape[1] != 64:
            raise ValueError(f'Expected 128-to-64 prior head; got {h.shape} -> {y.shape}')
        torch.save({'hidden': h[0].float().clone(),
                    'original_output': y[0].float().clone()}, filepath)
        captured[0] = True
        handle[0].remove()
        print('Saved first trained STRIVE prior-head activation:', filepath)

    handle[0] = target.register_forward_hook(on_first_forward)
    return handle[0]
