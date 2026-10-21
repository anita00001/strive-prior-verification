"""Stage-B verification for STRIVE's pretrained conditional prior.

Verified subnetwork:
    prior_net.msg[0].update_mlp -> prior_net.mlp_out

The edge-message computation and max aggregation are not included yet.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def _load_prefixed(module, state, prefix):
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise ValueError(f"No checkpoint keys found with prefix {prefix}")
    module.load_state_dict(sub, strict=True)


class StageBPrior(nn.Module):
    def __init__(self):
        super().__init__()
        # update input = node x(128) + aggregated message(128) + semantic one-hot(2)
        self.update = nn.Sequential(
            nn.Linear(258, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.output = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, u):
        return self.output(self.update(u))


def load_model(ckpt):
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = raw.get("model", raw.get("state_dict", raw))
    model = StageBPrior()
    _load_prefixed(model.update, state, "prior_net.msg.0.update_mlp.net.")
    _load_prefixed(model.output, state, "prior_net.mlp_out.net.")
    return model.eval()


def run(args):
    if args.epsilon <= 0 or args.delta <= 0:
        raise ValueError("epsilon and delta must be positive")

    model = load_model(args.ckpt)
    cap = torch.load(args.capture, map_location="cpu", weights_only=True)
    u0 = cap["update_input"].reshape(1, -1).float()
    recorded = cap["original_output"].reshape(1, -1).float()

    if u0.shape[1] != 258:
        raise RuntimeError(f"Expected 258-D Stage-B input, got {tuple(u0.shape)}")
    if recorded.shape[1] != 64:
        raise RuntimeError(f"Expected 64-D output, got {tuple(recorded.shape)}")

    with torch.no_grad():
        nominal = model(u0)

    max_abs_err = float((nominal - recorded).abs().max())
    if not torch.allclose(nominal, recorded, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            f"Wrapper does not reproduce captured STRIVE output "
            f"(max abs error={max_abs_err:.8g}). No verification claim made."
        )

    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    ptb = PerturbationLpNorm(norm=float("inf"), eps=args.epsilon)
    bounded_input = BoundedTensor(u0, ptb)
    bounded_model = BoundedModule(model, u0, device="cpu")

    if args.method == "CROWN-Optimized":
        bounded_model.set_bound_opts({"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}})

    lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method=args.method)
    lb, ub = lb.detach(), ub.detach()

    rows, results = [], {}
    certified = 0
    worst_name, worst_dev = None, -1.0

    for i in range(64):
        name = f"mu_{i}" if i < 32 else f"logvar_{i-32}"
        n = float(nominal[0, i])
        lo = float(lb[0, i])
        hi = float(ub[0, i])
        max_dev = max(n - lo, hi - n)
        ok = lo >= n - args.delta and hi <= n + args.delta
        verdict = "CERTIFIED" if ok else "NOT_PROVED"
        certified += int(ok)
        if max_dev > worst_dev:
            worst_dev, worst_name = max_dev, name

        row = {
            "output": name,
            "index": i,
            "nominal": n,
            "certified_lower": lo,
            "certified_upper": hi,
            "max_deviation": max_dev,
            "delta": args.delta,
            "verdict": verdict,
        }
        rows.append(row)
        results[name] = row
        print(
            f"{name:10s} nominal={n:+.6f} "
            f"interval=[{lo:+.6f}, {hi:+.6f}] "
            f"max_dev={max_dev:.6f} verdict={verdict}"
        )

    summary = {
        "stage": "B",
        "verified_component": "prior_net.msg[0].update_mlp -> prior_net.mlp_out",
        "excluded": [
            "prior_net.mlp_in", "edge_mlp", "max aggregation", "map encoder",
            "past encoder", "decoder", "vehicle dynamics"
        ],
        "input_dimension": 258,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "method": args.method,
        "wrapper_capture_max_abs_error": max_abs_err,
        "checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "num_outputs": 64,
        "num_certified": certified,
        "num_not_proved": 64 - certified,
        "all_certified": certified == 64,
        "worst_output_by_max_deviation": worst_name,
        "worst_max_deviation": worst_dev,
        "outputs": results,
        "interpretation": (
            "Local robustness of the update-MLP + output-MLP subnetwork around "
            "one captured real GNN update input. This is not verification of the complete prior GNN."
        ),
    }

    Path(args.out).write_text(json.dumps(summary, indent=2))
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n============================================================")
    print("STAGE B SUMMARY")
    print(f"Wrapper equivalence : max abs error = {max_abs_err:.8g}")
    print(f"Certified outputs   : {certified}/64")
    print(f"Not proved          : {64-certified}/64")
    print(f"Worst output        : {worst_name}")
    print(f"Worst max deviation : {worst_dev:.6f}")
    print(f"Allowed delta       : {args.delta:.6f}")
    print(f"Overall verdict     : {'ALL CERTIFIED' if certified == 64 else 'PARTIAL / NOT PROVED'}")
    print("============================================================")
    print("Wrote", args.out)
    print("Wrote", args.csv)
    return 0 if certified == 64 else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--capture", default="prior_stage_b_capture.pt")
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--method", choices=["IBP", "backward", "CROWN-Optimized"], default="backward")
    p.add_argument("--out", default="prior_stage_b_result.json")
    p.add_argument("--csv", default="prior_stage_b_result.csv")
    args = p.parse_args()
    try:
        raise SystemExit(run(args))
    except SystemExit:
        raise
    except Exception as exc:
        print("NO VERIFICATION RESULT:", type(exc).__name__, str(exc))
        raise SystemExit(3)
