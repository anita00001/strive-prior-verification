#!/usr/bin/env python3
"""
STRIVE Stage D1 target-only verifier.

Required artifacts:
    traffic_model.pth
    prior_stage_d_capture.pt

The capture is produced by capture_prior_stage_d.py from the first suitable
Stage-D scene. This verifier does not depend on Stage C.

Property:
    Only the selected target agent's 24 continuous state coordinates inside
    its 38-D past_encoder input are uncertain:

        4 timesteps x 6 state values = 24 uncertain values.

    For every output i=0,...,63:

        |y_i(x) - y_i(x0)| <= delta.

Held fixed:
    target visibility flags; target length/width attributes; target semantic
    tail; all map features; prior semantic labels; relative-pose edge features;
    graph topology; and all non-target agents.

This is an encoder-input certificate. It does not recompute map crops or
relative poses from perturbed world-frame histories.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def mlp(sizes):
    layers = [nn.Linear(sizes[0], sizes[1])]
    for i in range(1, len(sizes) - 1):
        layers.extend(
            [
                nn.LayerNorm(sizes[i]),
                nn.ReLU(),
                nn.Linear(sizes[i], sizes[i + 1]),
            ]
        )
    return nn.Sequential(*layers)


def load_prefixed(module, state, prefix):
    sub = {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }
    if not sub:
        raise ValueError("No checkpoint keys found with prefix %s" % prefix)
    module.load_state_dict(sub, strict=True)


def load_state(ckpt):
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    return raw.get("model", raw.get("state_dict", raw))


def output_name(i):
    return "mu_%d" % i if i < 32 else "logvar_%d" % (i - 32)


class FixedSceneTargetPrior(nn.Module):
    """Exact one-round target prior computation for the captured graph."""

    def __init__(self, sem, edge_index, rel_trans, target):
        super().__init__()

        self.mlp_in = mlp([130, 128, 128, 128])
        self.edge_mlp = mlp([264, 128, 128, 128])
        self.update_mlp = mlp([258, 128, 128])
        self.mlp_out = mlp([128, 128, 128, 64])

        sem = sem.float()
        edge_index = edge_index.long()
        rel_trans = rel_trans.float()

        if sem.dim() != 2 or sem.shape[1] != 2:
            raise RuntimeError("Expected sem [N,2], got %s" % (tuple(sem.shape),))
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise RuntimeError(
                "Expected edge_index [2,E], got %s" % (tuple(edge_index.shape),)
            )
        if rel_trans.dim() != 2 or rel_trans.shape[1] != 4:
            raise RuntimeError(
                "Expected rel_trans [E,4], got %s" % (tuple(rel_trans.shape),)
            )
        if rel_trans.shape[0] != edge_index.shape[1]:
            raise RuntimeError("rel_trans edge count does not match edge_index.")

        dst = edge_index[1]
        incoming = torch.nonzero(dst == int(target), as_tuple=False).flatten()
        if incoming.numel() == 0:
            raise RuntimeError("Captured target has no incoming edges.")

        source_nodes = edge_index[0, incoming]
        if bool((source_nodes == int(target)).any()):
            raise RuntimeError(
                "Target has an incoming self-loop; this verifier expects "
                "non-self incoming sources."
            )

        self.target = int(target)
        self.incoming_count = int(incoming.numel())

        self.register_buffer("sem", sem)
        self.register_buffer("source_nodes", source_nodes.long())
        self.register_buffer("rel_incoming", rel_trans[incoming].clone().float())

    @staticmethod
    def coordinate_max(messages):
        cur = messages[:, 0, :]
        for j in range(1, messages.shape[1]):
            nxt = messages[:, j, :]
            cur = torch.relu(cur - nxt) + nxt
        return cur

    def forward(self, prior_input):
        h = self.mlp_in(prior_input)

        h_t = h[:, self.target, :]
        h_s = h[:, self.source_nodes, :]

        B = prior_input.shape[0]
        K = int(self.source_nodes.numel())

        h_t_edges = h_t.unsqueeze(1).expand(-1, K, -1)
        sem_t = self.sem[self.target].reshape(1, 1, 2).expand(B, K, -1)
        sem_s = self.sem[self.source_nodes].unsqueeze(0).expand(B, -1, -1)
        rel = self.rel_incoming.unsqueeze(0).expand(B, -1, -1)

        msg_in = torch.cat([h_t_edges, h_s, sem_t, sem_s, rel], dim=-1)
        messages = self.edge_mlp(msg_in)
        aggr = self.coordinate_max(messages)

        update_in = torch.cat(
            [
                h_t,
                aggr,
                self.sem[self.target].reshape(1, 2).expand(B, -1),
            ],
            dim=-1,
        )

        updated = self.update_mlp(update_in)
        return self.mlp_out(updated)


class D1TargetOnlyModel(nn.Module):
    """Target past_encoder + fixed captured scene + target prior output."""

    def __init__(
        self,
        past_encoder,
        prior_model,
        fixed_node_input,
        fixed_target_past_input,
        target,
        state_indices,
    ):
        super().__init__()

        self.past_encoder = past_encoder
        self.prior_model = prior_model
        self.target = int(target)
        self.state_indices = [int(v) for v in state_indices]

        self.register_buffer(
            "fixed_node_input",
            fixed_node_input.unsqueeze(0).float(),
        )
        self.register_buffer(
            "fixed_target_past_input",
            fixed_target_past_input.reshape(1, 38).float(),
        )

        expected = []
        for t in range(4):
            expected.extend(range(t * 9, t * 9 + 6))
        if self.state_indices != expected:
            raise RuntimeError(
                "Expected STRIVE past input layout 4 x [state6,vis1,lw2] + sem2."
            )

    def reconstruct_target_past(self, state_root):
        B = state_root.shape[0]
        states = state_root.reshape(B, 4, 6)
        base = self.fixed_target_past_input.expand(B, -1)

        pieces = []
        for t in range(4):
            off = 9 * t
            fixed_tail = base[:, off + 6 : off + 9]
            pieces.append(torch.cat([states[:, t, :], fixed_tail], dim=-1))

        first36 = torch.cat(pieces, dim=-1)
        sem_tail = base[:, 36:38]
        return torch.cat([first36, sem_tail], dim=-1)

    def forward(self, state_root):
        B = state_root.shape[0]
        target_past = self.reconstruct_target_past(state_root)
        target_past_feat = self.past_encoder(target_past)

        prior_input = self.fixed_node_input.expand(B, -1, -1)
        target_fixed_tail = prior_input[:, self.target, 64:130]
        target_node = torch.cat([target_past_feat, target_fixed_tail], dim=-1)

        before = prior_input[:, : self.target, :]
        after = prior_input[:, self.target + 1 :, :]
        prior_input = torch.cat(
            [before, target_node.unsqueeze(1), after],
            dim=1,
        )

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

    cap = torch.load(args.capture, map_location="cpu", weights_only=True)

    required = {
        "target_past_input",
        "target_past_feat",
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

    target = int(cap["target_agent"])
    fixed_node_input = cap["node_input"].float()
    target_past_input = cap["target_past_input"].float()

    if fixed_node_input.dim() != 2 or fixed_node_input.shape[1] != 130:
        raise RuntimeError(
            "Expected node_input [N,130], got %s"
            % (tuple(fixed_node_input.shape),)
        )

    N = int(fixed_node_input.shape[0])
    if not (0 <= target < N):
        raise RuntimeError("target_agent outside captured node range.")
    if target_past_input.numel() != 38:
        raise RuntimeError("Expected target_past_input with 38 values.")

    state_indices = (
        cap["state_indices_in_past_input"].long().tolist()
        if "state_indices_in_past_input" in cap
        else [i for t in range(4) for i in range(t * 9, t * 9 + 6)]
    )

    past_encoder, prior_model = build_components(args.ckpt, cap)

    model = D1TargetOnlyModel(
        past_encoder,
        prior_model,
        fixed_node_input,
        target_past_input,
        target,
        state_indices,
    ).eval()

    state0 = target_past_input[state_indices].reshape(1, 24).float()

    recorded = cap["original_output"].float()
    if recorded.dim() == 3 and recorded.shape[0] == 1:
        recorded = recorded[0]
    if recorded.dim() != 2 or recorded.shape != (N, 64):
        raise RuntimeError(
            "Expected original_output [N,64], got %s" % (tuple(recorded.shape),)
        )
    recorded_target = recorded[target].reshape(1, 64)

    with torch.no_grad():
        reconstructed_past_feat = past_encoder(target_past_input.reshape(1, 38))
        nominal = model(state0)

    past_err = float(
        (
            reconstructed_past_feat
            - cap["target_past_feat"].reshape(1, 64).float()
        )
        .abs()
        .max()
    )
    full_err = float((nominal - recorded_target).abs().max())

    print("=" * 78)
    print("STRIVE STAGE D1: TARGET-ONLY PAST STATE")
    print("=" * 78)
    print("captured agents                 : %d" % N)
    print("target agent                    : %d" % target)
    print("incoming edges                  : %d" % prior_model.incoming_count)
    print("uncertain dimensions            : 24")
    print("epsilon                         : %.8g" % args.epsilon)
    print("delta                           : %.8g" % args.delta)
    print("past-encoder equivalence error  : %.8g" % past_err)
    print("full target-output error        : %.8g" % full_err)

    max_equiv = max(past_err, full_err)
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

    print("Building bounded D1 target-only model...")
    bounded = BoundedModule(model, state0, device="cpu")
    print("Computing %s bounds..." % args.method)

    if args.method == "CROWN-Optimized":
        bounded.set_bound_opts(
            {"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}}
        )

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
        "stage": "D1-target-only",
        "status": status,
        "scene_selection": cap.get("selection_rule", "first suitable Stage-D scene"),
        "captured_agents": N,
        "target_agent": target,
        "incoming_edges": int(prior_model.incoming_count),
        "uncertain_dimensions": 24,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "method": args.method,
        "fixed": [
            "all non-target agents",
            "target visibility flags",
            "target vehicle length/width attributes",
            "target past-encoder semantic tail",
            "all map features",
            "prior semantic labels",
            "relative-pose edge features",
            "graph topology",
        ],
        "equivalence": {
            "past_encoder_max_abs_error": past_err,
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
    print("STAGE D1 TARGET-ONLY SUMMARY")
    print("=" * 78)
    print("Status                         : %s" % status)
    print("Captured agents                : %d" % N)
    print("Target agent                   : %d" % target)
    print("Incoming edges                 : %d" % prior_model.incoming_count)
    print("Uncertain state dims           : 24")
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
        choices=["IBP", "backward", "CROWN-Optimized"],
        default="backward",
    )
    p.add_argument("--equiv-tol", type=float, default=1e-4)
    p.add_argument("--out", default="prior_stage_d1_result.json")
    p.add_argument("--csv", default="prior_stage_d1_result.csv")
    args = p.parse_args()

    try:
        raise SystemExit(main(args))
    except SystemExit:
        raise
    except Exception as exc:
        print("NO VERIFICATION RESULT:", type(exc).__name__, str(exc))
        raise SystemExit(3)
