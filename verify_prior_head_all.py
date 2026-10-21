"""Verify all 64 outputs of the original pretrained STRIVE conditional-prior output head.

Scope:
- Verifies prior_net.mlp_out only.
- Input is a captured 128-D hidden activation h0.
- Perturbation is an independent L-infinity box around h0.
- Outputs 0..31 are treated as mu_0..mu_31.
- Outputs 32..63 are treated as logvar_0..logvar_31.

This does NOT verify the upstream map encoder/GNN, downstream decoder,
vehicle dynamics, or complete STRIVE system.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def load_original_prior_head(ckpt_file):
    # Original STRIVE MLP([128,128,128,64]).
    layers = nn.Sequential(
        nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 64),
    )

    raw = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    state = raw.get("model", raw.get("state_dict", raw))

    prefix = "prior_net.mlp_out.net."
    sub = {
        key[len(prefix):]: tensor
        for key, tensor in state.items()
        if key.startswith(prefix)
    }

    if not sub:
        raise ValueError(f"No {prefix}* keys found: wrong STRIVE checkpoint?")

    layers.load_state_dict(sub, strict=True)
    return layers.eval()


def verify_all(
    checkpoint,
    capture,
    epsilon=1e-3,
    delta=0.05,
    method="backward",
    output_file="prior_result_all.json",
    csv_file="prior_result_all.csv",
):
    if not (epsilon > 0 and delta > 0):
        raise ValueError("epsilon and delta must be positive")

    head = load_original_prior_head(checkpoint)

    record = torch.load(capture, map_location="cpu", weights_only=True)
    h0 = record["hidden"].reshape(1, 128).float()
    recorded = record["original_output"].reshape(1, 64).float()

    with torch.no_grad():
        reference = head(h0)

    # Fail closed if the checkpoint and captured activation do not correspond.
    if not torch.allclose(reference, recorded, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "Original forward-hook output differs from reloaded pretrained head!"
        )

    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    perturbation = PerturbationLpNorm(norm=float("inf"), eps=epsilon)
    perturbed = BoundedTensor(h0, perturbation)

    bounded_head = BoundedModule(head, h0, device="cpu")

    if method == "CROWN-Optimized":
        bounded_head.set_bound_opts(
            {"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}}
        )

    lower, upper = bounded_head.compute_bounds(
        x=(perturbed,),
        method=method,
    )

    lower = lower.detach()
    upper = upper.detach()

    outcomes = {}
    csv_rows = []

    certified_count = 0
    worst_name = None
    worst_max_deviation = -1.0

    for idx in range(64):
        if idx < 32:
            name = f"mu_{idx}"
        else:
            name = f"logvar_{idx - 32}"

        base = float(reference[0, idx])
        lo = float(lower[0, idx])
        hi = float(upper[0, idx])

        lower_deviation = base - lo
        upper_deviation = hi - base
        max_deviation = max(lower_deviation, upper_deviation)

        success = lo >= base - delta and hi <= base + delta
        verdict = "CERTIFIED" if success else "NOT_PROVED"

        if success:
            certified_count += 1

        if max_deviation > worst_max_deviation:
            worst_max_deviation = max_deviation
            worst_name = name

        outcomes[name] = {
            "index": idx,
            "nominal": base,
            "certified_lower": lo,
            "certified_upper": hi,
            "lower_deviation": lower_deviation,
            "upper_deviation": upper_deviation,
            "max_deviation": max_deviation,
            "allowed_deviation": delta,
            "verdict": verdict,
        }

        csv_rows.append(
            {
                "output": name,
                "index": idx,
                "nominal": base,
                "certified_lower": lo,
                "certified_upper": hi,
                "max_deviation": max_deviation,
                "delta": delta,
                "verdict": verdict,
            }
        )

        print(
            f"{name:10s} "
            f"nominal={base:+.6f} "
            f"interval=[{lo:+.6f}, {hi:+.6f}] "
            f"max_dev={max_deviation:.6f} "
            f"verdict={verdict}"
        )

    sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

    result = {
        "component": "Original pretrained prior_net.mlp_out (MLP incl. LayerNorm)",
        "input_space": "Independent Linf perturbation of captured 128-D hidden activation",
        "checkpoint_sha256": sha256,
        "epsilon": epsilon,
        "delta": delta,
        "method": method,
        "num_outputs": 64,
        "num_certified": certified_count,
        "num_not_proved": 64 - certified_count,
        "all_certified": certified_count == 64,
        "worst_output_by_max_deviation": worst_name,
        "worst_max_deviation": worst_max_deviation,
        "scope_warning": (
            "No claim about complete prior GNN, map encoder, decoder, "
            "vehicle dynamics, or complete STRIVE."
        ),
        "outputs": outcomes,
    }

    Path(output_file).write_text(json.dumps(result, indent=2))

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output",
                "index",
                "nominal",
                "certified_lower",
                "certified_upper",
                "max_deviation",
                "delta",
                "verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print()
    print("============================================================")
    print(f"Certified outputs : {certified_count}/64")
    print(f"Not proved        : {64 - certified_count}/64")
    print(f"Worst output      : {worst_name}")
    print(f"Worst max dev.    : {worst_max_deviation:.6f}")
    print(f"Allowed delta     : {delta:.6f}")
    print(f"Overall verdict   : {'ALL CERTIFIED' if certified_count == 64 else 'PARTIAL / NOT PROVED'}")
    print("============================================================")
    print("Wrote", output_file)
    print("Wrote", csv_file)

    return certified_count == 64


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--capture", default="prior_head_capture.pt")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--method",
        choices=["IBP", "backward", "CROWN-Optimized"],
        default="backward",
    )
    parser.add_argument("--out", default="prior_result_all.json")
    parser.add_argument("--csv", default="prior_result_all.csv")

    args = parser.parse_args()

    try:
        ok = verify_all(
            args.ckpt,
            args.capture,
            args.epsilon,
            args.delta,
            args.method,
            args.out,
            args.csv,
        )
        # NOT_PROVED is not a counterexample. Bounds may simply be too loose.
        raise SystemExit(0 if ok else 2)
    except Exception as exc:
        print("NO VERIFICATION RESULT:", type(exc).__name__, str(exc))
        raise SystemExit(3)
