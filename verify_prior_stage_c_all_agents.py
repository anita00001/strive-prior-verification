"""Stage-C1 verifier for STRIVE's fixed-topology prior GNN.

This verifier strengthens the original target-only Stage-C property by
perturbing the continuous prior-input features of the selected target AND all
unique source agents that have an incoming edge to that target.

Verified scope:
- one captured scene
- fixed graph topology
- fixed relative-pose edge features
- fixed semantic labels
- selected target agent
- all unique agents that directly send a message to that target are uncertain
- for every involved agent, the first 128 prior-input dimensions
  (64 past features + 64 map features) are perturbed simultaneously
- the final two semantic dimensions remain fixed

Because STRIVE's prior uses one message-passing round, agents with no direct
incoming edge to the selected target cannot affect that target's prior output
in this round. They therefore do not need to be included in the bounded graph.
If the same source agent appears on multiple incoming edges, this verifier uses
one shared uncertain source input for all of those edges (no duplicate
independent copies).

The script refuses to make a verification claim unless the reduced wrapper
reproduces the captured STRIVE target output within the requested tolerance.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def mlp3(in_dim, hidden, out_dim):
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
    return nn.Sequential(
        nn.Linear(258, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )


def load_prefixed(module, state, prefix):
    sub = {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }
    if not sub:
        raise ValueError(f"No checkpoint keys found with prefix {prefix}")
    module.load_state_dict(sub, strict=True)


def stable_unique(values):
    """Return unique integer values in first-occurrence order."""
    out = []
    seen = set()
    for v in values:
        v = int(v)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class AllInvolvedPrior(nn.Module):
    """Exact selected-target one-round prior with uncertain source agents."""

    def __init__(
        self,
        sem_target,
        sem_sources_per_edge,
        rel_trans,
        source_positions_per_edge,
    ):
        super().__init__()

        self.mlp_in = mlp3(130, 128, 128)
        self.edge_mlp = mlp3(264, 128, 128)
        self.update = update_mlp()
        self.mlp_out = mlp3(128, 128, 64)

        # Constants for the captured topology / semantics / relative poses.
        self.register_buffer("sem_target", sem_target.reshape(1, 2).float())
        self.register_buffer(
            "sem_sources_per_edge",
            sem_sources_per_edge.unsqueeze(0).float(),
        )
        self.register_buffer(
            "rel_trans",
            rel_trans.unsqueeze(0).float(),
        )
        self.register_buffer(
            "source_positions_per_edge",
            source_positions_per_edge.long(),
        )

    @staticmethod
    def coordinate_max(messages):
        # messages: [1,K,128], K>=1.
        # Exact identity: max(a,b) = ReLU(a-b) + b.
        cur = messages[:, 0, :]
        for j in range(1, messages.shape[1]):
            nxt = messages[:, j, :]
            cur = torch.relu(cur - nxt) + nxt
        return cur

    def forward(self, involved_x):
        # involved_x: [1,N,130]
        # Position 0 = target; positions 1..N-1 = unique incoming sources.
        h_all = self.mlp_in(involved_x)  # [1,N,128]
        h_t = h_all[:, 0, :]  # [1,128]

        # Reuse the same hidden vector if one source occurs on multiple edges.
        h_sources = h_all[:, self.source_positions_per_edge, :]  # [1,K,128]

        k = h_sources.shape[1]
        h_i = h_t.unsqueeze(1).expand(-1, k, -1)
        sem_i = self.sem_target.unsqueeze(1).expand(-1, k, -1)

        msg_in = torch.cat(
            [
                h_i,
                h_sources,
                sem_i,
                self.sem_sources_per_edge,
                self.rel_trans,
            ],
            dim=-1,
        )  # [1,K,264]

        messages = self.edge_mlp(msg_in)  # [1,K,128]
        aggr = self.coordinate_max(messages)  # [1,128]

        update_in = torch.cat(
            [h_t, aggr, self.sem_target],
            dim=-1,
        )  # [1,258]
        updated = self.update(update_in)
        return self.mlp_out(updated)  # [1,64]


def load_checkpoint_state(path):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return raw.get("model", raw.get("state_dict", raw))


def build_model_and_input(ckpt, cap):
    state = load_checkpoint_state(ckpt)

    node_input = cap["node_input"].float()
    sem = cap["sem"].float()
    edge_index = cap["edge_index"].long()
    rel_trans_all = cap["rel_trans"].float()
    target = int(cap["target_agent"])

    src = edge_index[0]
    dst = edge_index[1]
    incoming_ids = torch.nonzero(dst == target, as_tuple=False).flatten()

    if incoming_ids.numel() == 0:
        raise RuntimeError(
            f"Target agent {target} has no incoming edges; "
            "this capture does not exercise GNN aggregation."
        )

    source_nodes_per_edge = src[incoming_ids]

    if bool((source_nodes_per_edge == target).any()):
        raise RuntimeError(
            "A self-loop enters the target agent. This C1 verifier currently "
            "expects incoming source agents to be distinct from the target."
        )

    unique_source_nodes = stable_unique(source_nodes_per_edge.tolist())
    source_to_position = {
        node_id: 1 + j for j, node_id in enumerate(unique_source_nodes)
    }
    source_positions_per_edge = torch.tensor(
        [source_to_position[int(v)] for v in source_nodes_per_edge.tolist()],
        dtype=torch.long,
    )

    involved_nodes = [target] + unique_source_nodes
    involved_x0 = node_input[involved_nodes].unsqueeze(0)  # [1,N,130]

    model = AllInvolvedPrior(
        sem_target=sem[target],
        sem_sources_per_edge=sem[source_nodes_per_edge],
        rel_trans=rel_trans_all[incoming_ids],
        source_positions_per_edge=source_positions_per_edge,
    )

    load_prefixed(model.mlp_in, state, "prior_net.mlp_in.net.")
    load_prefixed(model.edge_mlp, state, "prior_net.msg.0.edge_mlp.net.")
    load_prefixed(model.update, state, "prior_net.msg.0.update_mlp.net.")
    load_prefixed(model.mlp_out, state, "prior_net.mlp_out.net.")
    model.eval()

    recorded_target = cap["original_output"][target].reshape(1, 64).float()

    return (
        model,
        involved_x0,
        recorded_target,
        target,
        source_nodes_per_edge,
        unique_source_nodes,
        involved_nodes,
        incoming_ids,
    )


def main(args):
    if args.epsilon <= 0 or args.delta <= 0:
        raise ValueError("epsilon and delta must be positive")

    cap = torch.load(args.capture, map_location="cpu", weights_only=True)

    required = {
        "node_input",
        "sem",
        "edge_index",
        "rel_trans",
        "original_output",
        "target_agent",
    }
    missing = required.difference(cap)
    if missing:
        raise RuntimeError(f"Capture missing keys: {sorted(missing)}")

    (
        model,
        x0,
        recorded,
        target,
        source_nodes_per_edge,
        unique_source_nodes,
        involved_nodes,
        incoming_ids,
    ) = build_model_and_input(args.ckpt, cap)

    # Critical exactness/equivalence check before certification.
    with torch.no_grad():
        nominal = model(x0)

    err = (nominal - recorded).abs()
    max_abs_err = float(err.max())

    if not torch.allclose(
        nominal,
        recorded,
        atol=args.equiv_atol,
        rtol=args.equiv_rtol,
    ):
        worst = int(err.argmax())
        raise RuntimeError(
            "Stage-C1 all-involved-agents wrapper does not reproduce the "
            "captured STRIVE target output. "
            f"max_abs_error={max_abs_err:.8g}, worst_output={worst}. "
            "No verification claim made."
        )

    num_involved = len(involved_nodes)
    num_uncertain = num_involved * 128

    print("Stage-C1 all-involved-agents wrapper equivalence PASSED")
    print(f"  target agent             : {target}")
    print(f"  incoming edges           : {incoming_ids.numel()}")
    print(f"  unique source agents     : {len(unique_source_nodes)}")
    print(f"  involved agents total    : {num_involved}")
    print(f"  uncertain continuous dims: {num_uncertain}")
    print(f"  max abs error            : {max_abs_err:.8g}")

    # Perturb all continuous prior-input dimensions of every involved agent:
    # 64 past features + 64 map features = first 128 dimensions per agent.
    # Last two semantic dimensions remain fixed.
    x_L = x0.clone()
    x_U = x0.clone()
    x_L[:, :, :128] -= args.epsilon
    x_U[:, :, :128] += args.epsilon

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ptb = PerturbationLpNorm(
        norm=float("inf"),
        x_L=x_L,
        x_U=x_U,
    )
    bounded_x = BoundedTensor(x0, ptb)

    print("Building bounded Stage-C1 all-involved-agents model...")
    bounded_model = BoundedModule(model, x0, device="cpu")
    print(f"Computing {args.method} bounds...")

    if args.method == "CROWN-Optimized":
        bounded_model.set_bound_opts(
            {
                "optimize_bound_args": {
                    "iteration": args.optimized_iterations,
                    "lr_alpha": args.optimized_lr,
                }
            }
        )

    lb, ub = bounded_model.compute_bounds(
        x=(bounded_x,),
        method=args.method,
    )

    lb = lb.detach()
    ub = ub.detach()

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
        "stage": "C1",
        "verified_component": (
            "selected-target one-round prior GNN with all directly involved "
            "agents uncertain: mlp_in -> incoming edge_mlp messages -> "
            "coordinate max -> update_mlp -> mlp_out"
        ),
        "exact_reduction": (
            "Only the target and unique direct incoming source agents are "
            "included. This preserves the selected target function exactly "
            "for one message-passing round; unrelated scene agents cannot "
            "affect the target in that round. Repeated edges from one source "
            "share one uncertain source input."
        ),
        "target_agent": target,
        "incoming_edges_to_target": int(incoming_ids.numel()),
        "source_nodes_per_edge": [
            int(v) for v in source_nodes_per_edge.tolist()
        ],
        "unique_source_nodes": [int(v) for v in unique_source_nodes],
        "involved_nodes": [int(v) for v in involved_nodes],
        "num_involved_agents": num_involved,
        "uncertain_continuous_dimensions": num_uncertain,
        "perturbed": (
            "first 128 prior-input dimensions of target and every unique "
            "direct incoming source agent (64 past + 64 map features each)"
        ),
        "fixed": [
            "semantic features / final two prior-input dimensions",
            "relative-pose edge features",
            "graph topology",
        ],
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
            "semantic-label changes",
            "relative-pose perturbations",
            "graph-topology changes",
            "decoder",
            "vehicle dynamics",
        ],
        "outputs": outputs,
    }

    Path(args.out).write_text(json.dumps(result, indent=2))

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("============================================================")
    print("STAGE C1 ALL-INVOLVED-AGENTS SUMMARY")
    print(f"Target agent             : {target}")
    print(f"Incoming edges           : {incoming_ids.numel()}")
    print(f"Unique source agents     : {len(unique_source_nodes)}")
    print(f"Involved agents total    : {num_involved}")
    print(f"Uncertain continuous dims: {num_uncertain}")
    print(f"Wrapper equivalence      : max abs error = {max_abs_err:.8g}")
    print(f"Certified outputs        : {certified}/64")
    print(f"Not proved               : {64-certified}/64")
    print(f"Worst output             : {worst_name}")
    print(f"Worst max deviation      : {worst_dev:.6f}")
    print(f"Allowed delta            : {args.delta:.6f}")
    print(
        "Overall verdict          : "
        + ("ALL CERTIFIED" if certified == 64 else "PARTIAL / NOT PROVED")
    )
    print("============================================================")
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
    p.add_argument("--optimized-iterations", type=int, default=20)
    p.add_argument("--optimized-lr", type=float, default=0.1)
    p.add_argument("--equiv-atol", type=float, default=1e-4)
    p.add_argument("--equiv-rtol", type=float, default=1e-4)
    p.add_argument("--out", default="prior_stage_c1_all_agents_result.json")
    p.add_argument("--csv", default="prior_stage_c1_all_agents_result.csv")
    args = p.parse_args()

    try:
        raise SystemExit(main(args))
    except SystemExit:
        raise
    except Exception as exc:
        print("NO VERIFICATION RESULT:", type(exc).__name__, str(exc))
        raise SystemExit(3)
