#!/usr/bin/env python3
"""
STRIVE Stage D1 generalized all-agents verifier.

Required artifacts:
    traffic_model.pth
    prior_stage_d_capture.pt

The capture is produced by capture_prior_stage_d.py from the first suitable
Stage-D scene. This verifier does not depend on Stage C and does not assume a
fixed number of agents.

Property:
    For every captured agent, the 24 continuous state coordinates inside its
    38-D past_encoder input are perturbed jointly:

        4 timesteps x 6 state values = 24 values/agent.

    If the captured scene contains N agents, the verifier has N x 24 jointly
    uncertain variables. For every output i=0,...,63:

        |y_i(x) - y_i(x0)| <= delta.

Held fixed:
    visibility flags; vehicle length/width attributes; semantic tail inside
    past_encoder; all map features; prior semantic labels; relative-pose edge
    features; and graph topology.

This is not a world-frame history certificate: map crops and relative poses
are not recomputed after perturbing the past_encoder state coordinates.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from verify_prior_stage_d1 import (
    FixedSceneTargetPrior,
    load_prefixed,
    load_state,
    mlp,
    output_name,
)


class D1AllAgentsModel(nn.Module):
    """All-agent past_encoder + fixed map features + target prior output."""

    def __init__(
        self,
        past_encoder,
        prior_model,
        fixed_past_input,
        fixed_map_feat,
        fixed_sem,
        state_indices,
    ):
        super().__init__()

        self.past_encoder = past_encoder
        self.prior_model = prior_model

        self.register_buffer(
            "fixed_past_input",
            fixed_past_input.unsqueeze(0).float(),
        )
        self.register_buffer(
            "fixed_map_feat",
            fixed_map_feat.unsqueeze(0).float(),
        )
        self.register_buffer(
            "fixed_sem",
            fixed_sem.unsqueeze(0).float(),
        )

        self.num_agents = int(fixed_past_input.shape[0])
        self.state_indices = [int(v) for v in state_indices]

        expected = []
        for t in range(4):
            expected.extend(range(t * 9, t * 9 + 6))
        if self.state_indices != expected:
            raise RuntimeError(
                "Expected STRIVE past input layout 4 x [state6,vis1,lw2] + sem2."
            )

    def reconstruct_past_input(self, state_root):
        B = state_root.shape[0]
        states = state_root.reshape(B, self.num_agents, 4, 6)
        base = self.fixed_past_input.expand(B, -1, -1)

        steps = []
        for t in range(4):
            off = 9 * t
            fixed_tail = base[:, :, off + 6 : off + 9]
            steps.append(
                torch.cat([states[:, :, t, :], fixed_tail], dim=-1)
            )

        first36 = torch.cat(steps, dim=-1)
        sem_tail = base[:, :, 36:38]
        return torch.cat([first36, sem_tail], dim=-1)

    def forward(self, state_root):
        past_input = self.reconstruct_past_input(state_root)
        past_feat = self.past_encoder(past_input)

        B = state_root.shape[0]
        map_feat = self.fixed_map_feat.expand(B, -1, -1)
        sem = self.fixed_sem.expand(B, -1, -1)

        prior_input = torch.cat([past_feat, map_feat, sem], dim=-1)
        return self.prior_model(prior_input)


def build_components(ckpt, cap):
    state = load_state(ckpt)

    past_encoder = mlp([38, 128, 128, 128, 64])
    load_prefixed(past_encoder, state, "past_encoder.net.")

    prior_model = FixedSceneTargetPrior(
        cap["sem"].float(),
        cap["edge_index"].long(),
        cap["rel_trans"].float(),
        int(cap["target_agent"]),
    )

    load_prefixed(prior_model.mlp_in, state, "prior_net.mlp_in.net.")
    load_prefixed(prior_model.edge_mlp, state, "prior_net.msg.0.edge_mlp.net.")
    load_prefixed(
        prior_model.update_mlp,
        state,
        "prior_net.msg.0.update_mlp.net.",
    )
    load_prefixed(prior_model.mlp_out, state, "prior_net.mlp_out.net.")

    return past_encoder.eval(), prior_model.eval()


def main(args):
    if args.epsilon <= 0 or args.delta <= 0:
        raise ValueError("epsilon and delta must be positive")

    torch.set_num_threads(args.threads)

    cap = torch.load(args.capture, map_location="cpu", weights_only=True)

    required = {
        "all_past_input",
        "all_past_feat",
        "all_map_feat",
        "node_input",
        "sem",
        "edge_index",
        "rel_trans",
        "original_output",
        "target_agent",
    }
    missing = required.difference(cap)
    if missing:
        raise RuntimeError("Stage-D capture missing keys: %s" % sorted(missing))

    past_all = cap["all_past_input"].float()
    captured_past_feat = cap["all_past_feat"].float()
    fixed_map = cap["all_map_feat"].float()
    sem = cap["sem"].float()
    captured_node = cap["node_input"].float()
    target = int(cap["target_agent"])

    if past_all.dim() != 2 or past_all.shape[1] != 38:
        raise RuntimeError(
            "Expected all_past_input [N,38], got %s" % (tuple(past_all.shape),)
        )

    N = int(past_all.shape[0])

    if captured_past_feat.shape != (N, 64):
        raise RuntimeError("Expected all_past_feat [%d,64]." % N)
    if fixed_map.shape != (N, 64):
        raise RuntimeError("Expected all_map_feat [%d,64]." % N)
    if sem.shape != (N, 2):
        raise RuntimeError("Expected sem [%d,2]." % N)
    if captured_node.shape != (N, 130):
        raise RuntimeError("Expected node_input [%d,130]." % N)
    if not (0 <= target < N):
        raise RuntimeError("target_agent outside captured node range.")

    state_indices = (
        cap["state_indices_in_past_input"].long().tolist()
        if "state_indices_in_past_input" in cap
        else [i for t in range(4) for i in range(t * 9, t * 9 + 6)]
    )

    past_encoder, prior_model = build_components(args.ckpt, cap)

    model = D1AllAgentsModel(
        past_encoder,
        prior_model,
        past_all,
        fixed_map,
        sem,
        state_indices,
    ).eval()

    state0 = past_all[:, state_indices].unsqueeze(0).float()

    recorded = cap["original_output"].float()
    if recorded.dim() == 3 and recorded.shape[0] == 1:
        recorded = recorded[0]
    if recorded.dim() != 2 or recorded.shape != (N, 64):
        raise RuntimeError(
            "Expected original_output [N,64], got %s" % (tuple(recorded.shape),)
        )

    recorded_target = recorded[target].reshape(1, 64)

    with torch.no_grad():
        reconstructed_past = past_encoder(past_all)
        reconstructed_node = torch.cat(
            [reconstructed_past, fixed_map, sem],
            dim=-1,
        )
        nominal = model(state0)

    past_err = float(
        (reconstructed_past - captured_past_feat).abs().max()
    )
    node_err = float(
        (reconstructed_node - captured_node).abs().max()
    )
    full_err = float(
        (nominal - recorded_target).abs().max()
    )

    uncertain_dims = N * 24

    print("=" * 78)
    print("STRIVE STAGE D1-GENERALIZED: ALL CAPTURED AGENTS")
    print("=" * 78)
    print("captured agents                 : %d" % N)
    print("target agent                    : %d" % target)
    print("incoming edges                  : %d" % prior_model.incoming_count)
    print("state dims per agent            : 24")
    print("total uncertain dimensions      : %d" % uncertain_dims)
    print("epsilon                         : %.8g" % args.epsilon)
    print("delta                           : %.8g" % args.delta)
    print("past-encoder equivalence error  : %.8g" % past_err)
    print("node-input equivalence error    : %.8g" % node_err)
    print("full target-output error        : %.8g" % full_err)
    print(
        "fixed                           : visibility, lw, semantics, "
        "map features, relpose, topology"
    )

    max_equiv = max(past_err, node_err, full_err)
    if max_equiv > args.equiv_tol:
        raise RuntimeError(
            "Nominal equivalence failed (max %.8g > %.8g). No verification claim."
            % (max_equiv, args.equiv_tol)
        )

    x_L = state0 - float(args.epsilon)
    x_U = state0 + float(args.epsilon)

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L, x_U=x_U)
    bx = BoundedTensor(state0, ptb)

    print("Building bounded D1-generalized model...")
    bounded = BoundedModule(model, state0, device="cpu")
    print("Computing %s bounds..." % args.method)

    lb, ub = bounded.compute_bounds(x=(bx,), method=args.method)
    lb = lb.detach()
    ub = ub.detach()

    rows = []
    certified = 0
    worst_dev = -1.0
    worst_idx = -1

    for i in range(64):
        n = float(nominal[0, i])
        lo = float(lb[0, i])
        hi = float(ub[0, i])
        dev = max(n - lo, hi - n)
        ok = lo >= n - args.delta and hi <= n + args.delta
        verdict = "CERTIFIED" if ok else "NOT_PROVED"
        certified += int(ok)

        if dev > worst_dev:
            worst_dev = dev
            worst_idx = i

        row = {
            "output": output_name(i),
            "index": i,
            "nominal": n,
            "lower": lo,
            "upper": hi,
            "max_deviation": dev,
            "delta": args.delta,
            "verdict": verdict,
        }
        rows.append(row)

        print(
            "%-10s nominal=%+.6f interval=[%+.6f, %+.6f] "
            "max_dev=%.6f verdict=%s"
            % (output_name(i), n, lo, hi, dev, verdict)
        )

    status = "CERTIFIED" if certified == 64 else "NOT_PROVED"

    result = {
        "stage": "D1-generalized-all-agents",
        "status": status,
        "scene_selection": cap.get("selection_rule", "first suitable Stage-D scene"),
        "captured_agents": N,
        "target_agent": target,
        "incoming_edges": int(prior_model.incoming_count),
        "uncertain_dimensions": uncertain_dims,
        "per_agent_uncertain_dimensions": 24,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "method": args.method,
        "fixed": [
            "past visibility flags",
            "vehicle length/width attributes",
            "past-encoder semantic tail",
            "all map features",
            "prior semantic labels",
            "relative-pose edge features",
            "graph topology",
        ],
        "important_limitation": (
            "The uncertain variables are the state coordinates entering "
            "past_encoder. Map crops and relative poses are not recomputed "
            "from perturbed world-frame histories."
        ),
        "equivalence": {
            "past_encoder_max_abs_error": past_err,
            "node_input_max_abs_error": node_err,
            "full_target_output_max_abs_error": full_err,
        },
        "checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "num_certified": certified,
        "num_not_proved": 64 - certified,
        "worst_output": output_name(worst_idx),
        "worst_max_deviation": worst_dev,
        "outputs": rows,
    }

    Path(args.out).write_text(json.dumps(result, indent=2))

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("STAGE D1-GENERALIZED SUMMARY")
    print("=" * 78)
    print("Status                         : %s" % status)
    print("Captured agents                : %d" % N)
    print("Target agent                   : %d" % target)
    print("Incoming edges                 : %d" % prior_model.incoming_count)
    print("Uncertain state dims           : %d" % uncertain_dims)
    print("Certified outputs              : %d/64" % certified)
    print("Not proved                     : %d/64" % (64 - certified))
    print("Worst output                   : %s" % output_name(worst_idx))
    print("Worst max deviation            : %.6f" % worst_dev)
    print("=" * 78)
    print("Wrote", args.out)
    print("Wrote", args.csv)

    return 0 if certified == 64 else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument(
        "--method",
        choices=["IBP", "backward"],
        default="backward",
    )
    p.add_argument("--equiv-tol", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--out",
        default="prior_stage_d1_generalized_all_agents.json",
    )
    p.add_argument(
        "--csv",
        default="prior_stage_d1_generalized_all_agents.csv",
    )
    args = p.parse_args()

    try:
        raise SystemExit(main(args))
    except SystemExit:
        raise
    except Exception as exc:
        print("NO VERIFICATION RESULT:", type(exc).__name__, str(exc))
        raise SystemExit(3)
