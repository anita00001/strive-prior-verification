#!/usr/bin/env python3
"""
STRIVE Stage D1-generalized verifier.

This generalizes the previous target-only D1 certificate to ALL 16 involved
agents jointly.

Important scope
---------------
This mirrors the original D1 abstraction: the uncertain variables are the
24 continuous state coordinates inside each agent's 38-D *past-encoder input*,

    4 timesteps x 6 state values = 24 values/agent.

For 16 involved agents:
    16 x 24 = 384 jointly uncertain variables.

The following past-encoder inputs remain fixed:
  * visibility flags,
  * vehicle length/width attributes,
  * semantic tail inside the past encoder.

Also fixed in this D1-generalized stage:
  * all map features,
  * semantic labels used by the prior,
  * relative-pose edge features,
  * graph topology.

Therefore this is NOT yet a certificate for perturbing world-frame raw history
while simultaneously recomputing map crops and graph relative poses. It is the
correct all-agent generalization of the earlier D1 past-encoder-input property.

Property
--------
For all 384 state coordinates within +/- epsilon simultaneously,

    |y_i(x) - y_i(x0)| <= delta,  i=0,...,63.

Method
------
auto_LiRPA ordinary backward CROWN.

NOT_PROVED is not a counterexample.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from torch import nn

import verify_prior_stage_c_all_agents as c1
from verify_prior_stage_d1 import build_model as build_old_d1_model


STATE_INDICES_DEFAULT = []
for _t in range(4):
    STATE_INDICES_DEFAULT.extend(range(_t * 9, _t * 9 + 6))


class D1AllAgentsModel(nn.Module):
    def __init__(
        self,
        past_encoder,
        prior_model,
        fixed_past_input,
        fixed_map_feat,
        fixed_semantics,
        state_indices,
    ):
        super().__init__()
        self.past_encoder = past_encoder
        self.prior_model = prior_model

        self.register_buffer(
            "fixed_past_input",
            fixed_past_input.unsqueeze(0).float(),  # [1,N,38]
        )
        self.register_buffer(
            "fixed_map_feat",
            fixed_map_feat.unsqueeze(0).float(),  # [1,N,64]
        )
        self.register_buffer(
            "fixed_semantics",
            fixed_semantics.unsqueeze(0).float(),  # [1,N,2]
        )
        self.register_buffer(
            "state_indices",
            torch.tensor(state_indices, dtype=torch.long),
        )

        self.num_agents = int(fixed_past_input.shape[0])
        self.num_state = int(len(state_indices))

        if self.num_state != 24:
            raise RuntimeError("Expected 24 state coordinates per agent")

        # Build the complementary fixed-coordinate indices once.
        all_idx = list(range(38))
        state_set = set(int(i) for i in state_indices)
        fixed_idx = [i for i in all_idx if i not in state_set]
        self.register_buffer(
            "fixed_indices",
            torch.tensor(fixed_idx, dtype=torch.long),
        )

        # Instead of scatter/in-place assignment, retain exact input ordering by
        # rebuilding the known STRIVE layout per time step.
        expected = []
        for t in range(4):
            expected.extend(range(t * 9, t * 9 + 6))
        if list(state_indices) != expected:
            raise RuntimeError(
                "This verifier expects STRIVE layout "
                "4 x [state6,vis1,lw2] + sem2"
            )

    def reconstruct_past_input(self, state_root):
        # state_root: [B,N,24] = four contiguous 6-D state blocks.
        B = state_root.shape[0]
        states = state_root.reshape(B, self.num_agents, 4, 6)

        base = self.fixed_past_input.expand(B, -1, -1)

        steps = []
        for t in range(4):
            off = t * 9
            fixed_tail = base[:, :, off + 6 : off + 9]  # vis + lw2
            steps.append(torch.cat([states[:, :, t, :], fixed_tail], dim=-1))

        first36 = torch.cat(steps, dim=-1)
        sem_tail = base[:, :, 36:38]
        return torch.cat([first36, sem_tail], dim=-1)

    def forward(self, state_root):
        past_in = self.reconstruct_past_input(state_root)
        past_feat = self.past_encoder(past_in)

        B = state_root.shape[0]
        map_feat = self.fixed_map_feat.expand(B, -1, -1)
        sem = self.fixed_semantics.expand(B, -1, -1)

        prior_in = torch.cat([past_feat, map_feat, sem], dim=-1)
        return self.prior_model(prior_in)


def output_name(i):
    return "mu_%d" % i if i < 32 else "logvar_%d" % (i - 32)


def extract_state_root(past_input, state_indices):
    return past_input[:, state_indices].unsqueeze(0)


def main(args):
    torch.set_num_threads(args.threads)

    ccap = torch.load(args.c_capture, map_location="cpu", weights_only=True)
    dcap = torch.load(args.d1_all_capture, map_location="cpu", weights_only=True)
    old_dcap = torch.load(args.old_d_capture, map_location="cpu", weights_only=True)

    (
        prior_model,
        involved_x0,
        recorded,
        target,
        source_nodes_per_edge,
        unique_source_nodes,
        involved_nodes,
        incoming_ids,
    ) = c1.build_model_and_input(args.ckpt, ccap)

    prior_model = prior_model.eval()

    # Reuse the already-debugged old D1 model builder only to recover the
    # checkpoint's trained past_encoder module.
    old_full_model, _, _ = build_old_d1_model(args.ckpt, old_dcap)
    past_encoder = copy.deepcopy(old_full_model.past_encoder).float().eval()

    if "all_past_input" not in dcap or "all_past_feat" not in dcap:
        raise RuntimeError(
            "D1-all capture must contain all_past_input and all_past_feat"
        )

    past_all = dcap["all_past_input"].float()
    past_feat_all = dcap["all_past_feat"].float()

    state_indices = (
        dcap["state_indices_in_past_input"].long().tolist()
        if "state_indices_in_past_input" in dcap
        else list(STATE_INDICES_DEFAULT)
    )

    # C1 involved-node order is [target, unique direct sources...].
    idx = torch.tensor(involved_nodes, dtype=torch.long)
    past_involved = past_all[idx]
    captured_past_feat = past_feat_all[idx]

    N = len(involved_nodes)
    if N != 16:
        raise RuntimeError(
            "Expected 16 involved agents for D1-generalized, got %d" % N
        )

    if past_involved.shape != (N, 38):
        raise RuntimeError(
            "Expected involved past input shape (%d,38), got %s"
            % (N, tuple(past_involved.shape))
        )

    # Fixed values must come from the exact C-stage scene.
    fixed_map = involved_x0[0, :, 64:128].clone().float()
    fixed_sem = involved_x0[0, :, 128:130].clone().float()
    c_past_feat = involved_x0[0, :, 0:64].clone().float()

    model = D1AllAgentsModel(
        past_encoder,
        prior_model,
        past_involved,
        fixed_map,
        fixed_sem,
        state_indices,
    ).eval()

    state0 = extract_state_root(past_involved, state_indices).float()

    # ------------------------------------------------------------
    # Nominal equivalence checks before making any formal claim.
    # ------------------------------------------------------------
    with torch.no_grad():
        reconstructed_past = past_encoder(past_involved)
        nominal = model(state0)

    past_capture_err = float(
        (reconstructed_past - captured_past_feat).abs().max()
    )
    past_c_err = float(
        (reconstructed_past - c_past_feat).abs().max()
    )
    full_err = float(
        (nominal - recorded.float()).abs().max()
    )

    print("=" * 78)
    print("STRIVE STAGE D1-GENERALIZED: ALL-AGENT PAST STATES")
    print("=" * 78)
    print("target agent                    : %d" % target)
    print("incoming edges                  : %d" % int(incoming_ids.numel()))
    print("involved agents                 : %d" % N)
    print("state dims per agent            : 24")
    print("total uncertain dimensions      : %d" % (N * 24))
    print("epsilon                         : %.8g" % args.epsilon)
    print("delta                           : %.8g" % args.delta)
    print("past-encoder/capture error      : %.8g" % past_capture_err)
    print("past-feature/C-stage error      : %.8g" % past_c_err)
    print("full target-output error        : %.8g" % full_err)
    print("fixed                           : visibility, lw, semantics, map, relpose, topology")

    max_equiv = max(past_capture_err, past_c_err, full_err)
    if max_equiv > args.equiv_tol:
        raise RuntimeError(
            "Nominal equivalence failed (max %.8g > %.8g). "
            "No verification claim." % (max_equiv, args.equiv_tol)
        )

    # ------------------------------------------------------------
    # Formal root box: all 384 state coordinates perturbed jointly.
    # ------------------------------------------------------------
    x_L = state0 - float(args.epsilon)
    x_U = state0 + float(args.epsilon)

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ptb = PerturbationLpNorm(
        norm=float("inf"),
        x_L=x_L,
        x_U=x_U,
    )
    bx = BoundedTensor(state0, ptb)

    print("Building bounded D1-generalized model...")
    bounded = BoundedModule(model, state0, device="cpu")
    print("Computing %s bounds..." % args.method)

    lb, ub = bounded.compute_bounds(
        x=(bx,),
        method=args.method,
    )
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
        ok = (lo >= n - args.delta) and (hi <= n + args.delta)
        verdict = "CERTIFIED" if ok else "NOT_PROVED"
        certified += int(ok)

        if dev > worst_dev:
            worst_dev = dev
            worst_idx = i

        rows.append(
            {
                "output": output_name(i),
                "index": i,
                "nominal": n,
                "lower": lo,
                "upper": hi,
                "max_deviation": dev,
                "delta": args.delta,
                "verdict": verdict,
            }
        )

        print(
            "%-10s nominal=%+.6f interval=[%+.6f, %+.6f] "
            "max_dev=%.6f verdict=%s"
            % (output_name(i), n, lo, hi, dev, verdict)
        )

    status = "CERTIFIED" if certified == 64 else "NOT_PROVED"

    print()
    print("=" * 78)
    print("STAGE D1-GENERALIZED SUMMARY")
    print("=" * 78)
    print("Status                         : %s" % status)
    print("Target agent                   : %d" % target)
    print("Involved agents                : %d" % N)
    print("Incoming edges                 : %d" % int(incoming_ids.numel()))
    print("Uncertain state dims           : %d" % (N * 24))
    print("Past steps x state             : 4 x 6 per agent")
    print("Epsilon                        : %.8g" % args.epsilon)
    print("Delta                          : %.8g" % args.delta)
    print("Certified outputs              : %d/64" % certified)
    print("Not proved                     : %d/64" % (64 - certified))
    print("Worst output                   : %s" % output_name(worst_idx))
    print("Worst max deviation            : %.6f" % worst_dev)
    print(
        "Overall verdict                 : %s"
        % ("ALL CERTIFIED" if certified == 64 else "PARTIAL / NOT PROVED")
    )
    print("=" * 78)

    result = {
        "stage": "D1-generalized-all-agents",
        "status": status,
        "scope": {
            "target_agent": target,
            "involved_agents": N,
            "incoming_edges": int(incoming_ids.numel()),
            "uncertain_dimensions": N * 24,
            "per_agent": "4 timesteps x 6 state coordinates in 38-D past-encoder input",
            "epsilon": args.epsilon,
            "delta": args.delta,
            "fixed": [
                "past visibility flags",
                "vehicle length/width attributes",
                "past-encoder semantic tail",
                "map features",
                "prior semantic labels",
                "relative-pose edge features",
                "graph topology",
            ],
            "important_limitation": (
                "This perturbs the state coordinates entering the past encoder, "
                "matching the original D1 abstraction. It does not recompute map "
                "crops or relative poses from perturbed world-frame history."
            ),
        },
        "equivalence": {
            "past_encoder_vs_d1_all_capture": past_capture_err,
            "past_feature_vs_c_capture": past_c_err,
            "full_output_vs_c_capture": full_err,
        },
        "method": args.method,
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

    print("Wrote", args.out)
    print("Wrote", args.csv)

    return 0 if certified == 64 else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--c-capture", required=True)
    p.add_argument("--d1-all-capture", required=True)
    p.add_argument(
        "--old-d-capture",
        required=True,
        help="Existing prior_stage_d_capture.pt; used only to recover past_encoder via old D1 build_model.",
    )
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
