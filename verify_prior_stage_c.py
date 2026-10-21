"""Stage C: fixed-scene verification of the full STRIVE conditional-prior GNN.

Included:
    prior_net.mlp_in
    prior_net.msg[0].edge_mlp
    coordinate-wise max aggregation (exact ReLU rewrite)
    prior_net.msg[0].update_mlp
    prior_net.mlp_out

Fixed:
    graph topology (edge_index)
    relative-pose edge features
    semantic one-hot features used by message/update functions
    all non-target agents

Perturbed:
    first 128 continuous dimensions of one target agent's prior input:
        64-D past feature + 64-D map feature
    The last 2 semantic dimensions in prior input remain fixed.

Property for target agent:
    ||x_target - x_target,0||_inf <= epsilon
        =>
    |y_i(x) - y_i(x0)| <= delta, for all 64 prior outputs.

This is a fixed-scene prior-GNN property. It does not yet verify the upstream
past encoder or map CNN/GroupNorm that produced the captured features.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def mlp_3(in_dim, hidden, out_dim):
    # STRIVE MLP([in_dim, hidden, hidden, out_dim])
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def update_mlp():
    # STRIVE MLP([258, 128, 128])
    return nn.Sequential(
        nn.Linear(258, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )


def load_prefixed(module, state, prefix):
    sub = {
        key[len(prefix):]: tensor
        for key, tensor in state.items()
        if key.startswith(prefix)
    }
    if not sub:
        raise ValueError(f"No checkpoint keys found with prefix {prefix}")
    module.load_state_dict(sub, strict=True)


class FixedScenePriorGNN(nn.Module):
    def __init__(self, edge_index, sem, rel_trans, target_agent):
        super().__init__()

        self.mlp_in = mlp_3(130, 128, 128)
        self.edge_mlp = mlp_3(264, 128, 128)
        self.update = update_mlp()
        self.mlp_out = mlp_3(128, 128, 64)

        self.target_agent = int(target_agent)
        self.num_nodes = int(sem.shape[0])

        # Leading size-1 dimension is the verification batch dimension.
        self.register_buffer("sem", sem.float().unsqueeze(0))
        self.register_buffer("rel_trans", rel_trans.float().unsqueeze(0))
        self.register_buffer("src", edge_index[0].long())
        self.register_buffer("dst", edge_index[1].long())
        self.register_buffer("zero_message", torch.zeros(1, 128))

        # Incoming edge IDs are fixed because this is fixed-scene verification.
        # Register them as buffers so tracing treats them as model constants.
        for node in range(self.num_nodes):
            ids = torch.nonzero(self.dst == node, as_tuple=False).flatten()
            self.register_buffer(f"incoming_{node}", ids)

    @staticmethod
    def pairwise_max_relu(values):
        """Exact coordinate-wise max using only ReLU + affine operations.

        max(a,b) = ReLU(a-b) + b.

        values: [1, K, 128], K >= 1
        returns: [1, 128]
        """
        cur = values[:, 0, :]
        for j in range(1, values.shape[1]):
            nxt = values[:, j, :]
            cur = torch.relu(cur - nxt) + nxt
        return cur

    def full_forward(self, x):
        # x: [1, N, 130]
        h = self.mlp_in(x)  # [1,N,128]

        x_i = torch.index_select(h, 1, self.dst)  # targets
        x_j = torch.index_select(h, 1, self.src)  # sources

        sem_i = torch.index_select(self.sem, 1, self.dst)
        sem_j = torch.index_select(self.sem, 1, self.src)

        # Exact STRIVE order:
        # [x_i, x_j, sem_i, sem_j, rel_trans]
        msg_in = torch.cat(
            [x_i, x_j, sem_i, sem_j, self.rel_trans],
            dim=-1,
        )
        messages = self.edge_mlp(msg_in)  # [1,E,128]

        aggregated = []
        for node in range(self.num_nodes):
            ids = getattr(self, f"incoming_{node}")
            if ids.numel() == 0:
                aggregated.append(self.zero_message)
            else:
                incoming = torch.index_select(messages, 1, ids)
                aggregated.append(self.pairwise_max_relu(incoming))

        aggr = torch.stack(aggregated, dim=1)  # [1,N,128]

        update_in = torch.cat([h, aggr, self.sem], dim=-1)  # [1,N,258]
        updated = self.update(update_in)
        out = self.mlp_out(updated)  # [1,N,64]
        return out

    def forward(self, x):
        # Bound only the selected agent's 64 outputs, reducing verification cost.
        out = self.full_forward(x)
        return out[:, self.target_agent, :]


def build_model(checkpoint, capture):
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("model", raw.get("state_dict", raw))

    model = FixedScenePriorGNN(
        capture["edge_index"],
        capture["sem"],
        capture["rel_trans"],
        capture["target_agent"],
    )

    load_prefixed(model.mlp_in, state, "prior_net.mlp_in.net.")
    load_prefixed(model.edge_mlp, state, "prior_net.msg.0.edge_mlp.net.")
    load_prefixed(model.update, state, "prior_net.msg.0.update_mlp.net.")
    load_prefixed(model.mlp_out, state, "prior_net.mlp_out.net.")

    return model.eval()


def main(args):
    if args.epsilon <= 0 or args.delta <= 0:
        raise ValueError("epsilon and delta must be positive")

    cap = torch.load(args.capture, map_location="cpu", weights_only=True)

    required = {
        "node_input", "sem", "edge_index", "rel_trans",
        "original_output", "target_agent"
    }
    missing = required.difference(cap)
    if missing:
        raise RuntimeError(f"Capture is missing keys: {sorted(missing)}")

    x0_nodes = cap["node_input"].float()
    recorded = cap["original_output"].float()

    if x0_nodes.ndim != 2 or x0_nodes.shape[1] != 130:
        raise RuntimeError(
            f"Expected node_input [N,130], got {tuple(x0_nodes.shape)}"
        )
    if recorded.shape != (x0_nodes.shape[0], 64):
        raise RuntimeError(
            f"Expected original_output [N,64], got {tuple(recorded.shape)}"
        )
    if cap["sem"].shape != (x0_nodes.shape[0], 2):
        raise RuntimeError(
            f"Expected sem [N,2], got {tuple(cap['sem'].shape)}"
        )
    if cap["edge_index"].ndim != 2 or cap["edge_index"].shape[0] != 2:
        raise RuntimeError("edge_index must have shape [2,E]")
    if cap["rel_trans"].shape != (cap["edge_index"].shape[1], 4):
        raise RuntimeError(
            "rel_trans must have shape [E,4] matching edge_index."
        )

    target = int(cap["target_agent"])
    if not (0 <= target < x0_nodes.shape[0]):
        raise RuntimeError(f"Invalid target_agent={target}")

    model = build_model(args.ckpt, cap)
    x0 = x0_nodes.unsqueeze(0)  # [1,N,130]

    # Before doing any certification, require our verifier-friendly fixed-graph
    # implementation to reproduce the original trained STRIVE prior_net.
    with torch.no_grad():
        wrapper_full = model.full_forward(x0)[0]

    abs_err = (wrapper_full - recorded).abs()
    max_abs_err = float(abs_err.max())
    if not torch.allclose(
        wrapper_full, recorded, atol=args.equiv_atol, rtol=args.equiv_rtol
    ):
        worst_flat = int(abs_err.argmax())
        worst_node = worst_flat // 64
        worst_output = worst_flat % 64
        raise RuntimeError(
            "Fixed-scene wrapper does not reproduce original STRIVE prior_net. "
            f"max_abs_error={max_abs_err:.8g}, "
            f"worst_node={worst_node}, worst_output={worst_output}. "
            "No verification claim made."
        )

    nominal = wrapper_full[target].reshape(1, 64)

    # Only the target agent's continuous prior features are perturbed:
    # [past_feat(64), map_feat(64)]. Semantic x[:, :, 128:130] stays fixed.
    x_L = x0.clone()
    x_U = x0.clone()
    x_L[:, target, :128] -= args.epsilon
    x_U[:, target, :128] += args.epsilon

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ptb = PerturbationLpNorm(
        norm=float("inf"),
        x_L=x_L,
        x_U=x_U,
    )
    bounded_x = BoundedTensor(x0, ptb)

    bounded_model = BoundedModule(model, x0, device="cpu")

    if args.method == "CROWN-Optimized":
        bounded_model.set_bound_opts(
            {"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}}
        )

    lb, ub = bounded_model.compute_bounds(
        x=(bounded_x,),
        method=args.method,
    )
    lb, ub = lb.detach(), ub.detach()

    rows = []
    outputs = {}
    certified = 0
    worst_name = None
    worst_dev = -1.0

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
            worst_dev = max_dev
            worst_name = name

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
        outputs[name] = row

        print(
            f"{name:10s} nominal={n:+.6f} "
            f"interval=[{lo:+.6f}, {hi:+.6f}] "
            f"max_dev={max_dev:.6f} verdict={verdict}"
        )

    result = {
        "stage": "C",
        "verified_component": (
            "fixed-scene full prior_net: mlp_in -> edge_mlp -> "
            "max aggregation -> update_mlp -> mlp_out"
        ),
        "fixed": [
            "edge_index",
            "relative-pose edge features",
            "semantic class features",
            "all non-target agents",
        ],
        "perturbed": (
            "target agent's first 128 prior-input dimensions "
            "(64 past-feature + 64 map-feature dimensions)"
        ),
        "target_agent": target,
        "num_nodes": int(x0_nodes.shape[0]),
        "num_edges": int(cap["edge_index"].shape[1]),
        "input_feature_dim": 130,
        "perturbed_feature_dim": 128,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "method": args.method,
        "wrapper_capture_max_abs_error": max_abs_err,
        "checkpoint_sha256": hashlib.sha256(
            Path(args.ckpt).read_bytes()
        ).hexdigest(),
        "num_outputs": 64,
        "num_certified": certified,
        "num_not_proved": 64 - certified,
        "all_certified": certified == 64,
        "worst_output_by_max_deviation": worst_name,
        "worst_max_deviation": worst_dev,
        "excluded": [
            "past encoder",
            "map CNN",
            "GroupNorm",
            "decoder",
            "vehicle dynamics",
        ],
        "outputs": outputs,
        "interpretation": (
            "Local robustness of the complete pretrained STRIVE prior GNN "
            "for one fixed captured scene and one selected agent. Graph "
            "topology, relative poses, semantic labels, and other agents are "
            "held fixed. This is not full conditional-prior verification."
        ),
    }

    Path(args.out).write_text(json.dumps(result, indent=2))
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("============================================================")
    print("STAGE C SUMMARY")
    print(f"Scene nodes         : {x0_nodes.shape[0]}")
    print(f"Scene edges         : {cap['edge_index'].shape[1]}")
    print(f"Target agent        : {target}")
    print(f"Wrapper equivalence : max abs error = {max_abs_err:.8g}")
    print(f"Certified outputs   : {certified}/64")
    print(f"Not proved          : {64-certified}/64")
    print(f"Worst output        : {worst_name}")
    print(f"Worst max deviation : {worst_dev:.6f}")
    print(f"Allowed delta       : {args.delta:.6f}")
    print(
        "Overall verdict     : "
        + ("ALL CERTIFIED" if certified == 64 else "PARTIAL / NOT PROVED")
    )
    print("============================================================")
    print("Wrote", args.out)
    print("Wrote", args.csv)

    return 0 if certified == 64 else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--capture", default="prior_stage_c_capture.pt")
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument(
        "--method",
        choices=["IBP", "backward", "CROWN-Optimized"],
        default="backward",
    )
    p.add_argument("--equiv-atol", type=float, default=1e-4)
    p.add_argument("--equiv-rtol", type=float, default=1e-4)
    p.add_argument("--out", default="prior_stage_c_result.json")
    p.add_argument("--csv", default="prior_stage_c_result.csv")
    args = p.parse_args()

    try:
        raise SystemExit(main(args))
    except SystemExit:
        raise
    except Exception as exc:
        print("NO VERIFICATION RESULT:", type(exc).__name__, str(exc))
        raise SystemExit(3)
