#!/usr/bin/env python3
"""
Unified Stage-D capture helper for STRIVE D1 verification.

This version does NOT depend on Stage C. It captures the first suitable graph
encountered by test_traffic.py and writes one artifact:

    prior_stage_d_capture.pt

That single capture is consumed by both:

    verify_prior_stage_d1.py
    verify_prior_stage_d1_generalized_all_agents.py

A graph is suitable when it contains a target with at least one incoming
prior-GNN edge and no self-loop among that target's incoming edges. The first
such graph is captured, and the first suitable target in that graph is used.

Expected past_encoder input layout:
    4 x [state6, visibility1, vehicle_attributes2] + semantic2 = 38 dims

D1 perturbs only the 24 continuous state coordinates:
    4 timesteps x 6 state values = 24 values/agent.

Integration in STRIVE/src/test_traffic.py:

    from capture_prior_stage_d import capture_stage_d

Immediately after the normal forward inside run_one_epoch():

    pred = model(scene_graph, map_idx, map_env, use_post_mean=True)
    capture_stage_d(
        model,
        scene_graph,
        map_idx,
        map_env,
        output_capture=os.path.expanduser(
            "~/formal-verification-workspace/strive-prior-verification/"
            "prior_stage_d_capture.pt"
        ),
    )

The helper exits with SystemExit(0) after a successful capture.
"""

from __future__ import print_function

import os
import torch


_DONE = False


def _state_indices():
    idx = []
    for t in range(4):
        base = 9 * t
        idx.extend(range(base, base + 6))
    return torch.tensor(idx, dtype=torch.long)


def _graph_ranges(scene_graph, total_nodes):
    if hasattr(scene_graph, "ptr") and scene_graph.ptr is not None:
        ptr = scene_graph.ptr.detach().cpu().long()
        return [
            (g, int(ptr[g]), int(ptr[g + 1]))
            for g in range(len(ptr) - 1)
        ]
    return [(0, 0, int(total_nodes))]


def _choose_first_suitable_graph_and_target(edge_index, graph_ranges):
    src_all = edge_index[0]
    dst_all = edge_index[1]

    for graph_id, start, end in graph_ranges:
        mask = (
            (src_all >= start)
            & (src_all < end)
            & (dst_all >= start)
            & (dst_all < end)
        )
        if not bool(mask.any()):
            continue

        local_edges = edge_index[:, mask] - start
        local_nodes = end - start

        for target in range(local_nodes):
            incoming = torch.nonzero(
                local_edges[1] == target,
                as_tuple=False,
            ).flatten()
            if incoming.numel() == 0:
                continue

            sources = local_edges[0, incoming]
            if bool((sources == target).any()):
                continue

            return {
                "graph_id": graph_id,
                "start": start,
                "end": end,
                "edge_mask": mask,
                "local_edges": local_edges,
                "target": target,
                "incoming": incoming,
                "sources": sources,
            }

    return None


def capture_stage_d(
    model,
    scene_graph,
    map_idx,
    map_env,
    output_capture="./prior_stage_d_capture.pt",
):
    """Capture the first suitable Stage-D scene."""
    global _DONE

    if _DONE:
        return True

    output_capture = os.path.expanduser(output_capture)

    if os.path.exists(output_capture):
        print("[Stage-D capture] Output already exists:", output_capture)
        print(
            "[Stage-D capture] Remove or rename it only if you intentionally "
            "want to recapture."
        )
        raise SystemExit(0)

    if getattr(model, "traj_encoder_type", None) != "mlp":
        raise RuntimeError(
            "Stage-D D1 capture currently expects STRIVE's MLP past encoder. "
            "Found traj_encoder_type=%r"
            % getattr(model, "traj_encoder_type", None)
        )

    captured = {}

    def past_pre_hook(module, inputs):
        if "all_past_input" not in captured:
            captured["all_past_input"] = inputs[0].detach().cpu().clone()

    def map_pre_hook(module, inputs):
        if "all_map_obs" not in captured:
            captured["all_map_obs"] = inputs[0].detach().cpu().clone()

    h_past = model.past_encoder.register_forward_pre_hook(past_pre_hook)
    h_map = model.map_conv.register_forward_pre_hook(map_pre_hook)

    try:
        with torch.no_grad():
            scene_graph.pos = scene_graph.past[:, -1, :4]
            map_feat = model.encode_map(scene_graph, map_idx, map_env)
            past_feat = model.encode_past(scene_graph)
    finally:
        h_past.remove()
        h_map.remove()

    if "all_past_input" not in captured:
        raise RuntimeError("Failed to capture past_encoder input.")
    if "all_map_obs" not in captured:
        raise RuntimeError("Failed to capture map_conv input.")

    all_past_input = captured["all_past_input"]
    all_map_obs = captured["all_map_obs"]
    total_nodes = int(past_feat.shape[0])

    if all_past_input.dim() != 2 or all_past_input.shape != (total_nodes, 38):
        raise RuntimeError(
            "Expected past_encoder input [%d,38], got %s"
            % (total_nodes, tuple(all_past_input.shape))
        )

    if (
        all_map_obs.dim() != 4
        or all_map_obs.shape[0] != total_nodes
        or all_map_obs.shape[1] != 4
    ):
        raise RuntimeError(
            "Expected map CNN input [N,4,H,W], got %s"
            % (tuple(all_map_obs.shape),)
        )

    edge_index_all = scene_graph.edge_index.detach().cpu().long()
    ranges = _graph_ranges(scene_graph, total_nodes)
    chosen = _choose_first_suitable_graph_and_target(edge_index_all, ranges)
    if chosen is None:
        return False

    start = chosen["start"]
    end = chosen["end"]
    mask = chosen["edge_mask"]
    local_edges = chosen["local_edges"]
    target = chosen["target"]
    incoming = chosen["incoming"]
    sources = chosen["sources"]

    sem_all = scene_graph.sem.detach().cpu()
    node_input_all = torch.cat(
        [past_feat.detach().cpu(), map_feat.detach().cpu(), sem_all],
        dim=-1,
    )
    if node_input_all.shape[1] != 130:
        raise RuntimeError(
            "Expected prior node input dim 130, got %d"
            % node_input_all.shape[1]
        )

    edge_capture = {}
    prior_capture = {}

    def edge_pre_hook(module, inputs):
        if "rel_trans_all" in edge_capture:
            return
        msg_in = inputs[0].detach().cpu()
        if msg_in.dim() != 2 or msg_in.shape[1] < 4:
            raise RuntimeError(
                "Unexpected edge_mlp input shape %s"
                % (tuple(msg_in.shape),)
            )
        edge_capture["rel_trans_all"] = msg_in[:, -4:].clone()

    def prior_post_hook(module, inputs, output):
        if "prior_output_all" not in prior_capture:
            prior_capture["prior_output_all"] = output.detach().cpu().clone()

    h_edge = model.prior_net.msg[0].edge_mlp.register_forward_pre_hook(edge_pre_hook)
    h_prior = model.prior_net.register_forward_hook(prior_post_hook)

    try:
        with torch.no_grad():
            scene_graph.pos = scene_graph.past[:, -1, :4]
            model.prior(scene_graph, map_feat, past_feat)
    finally:
        h_edge.remove()
        h_prior.remove()

    if "rel_trans_all" not in edge_capture:
        raise RuntimeError("Failed to capture edge relative transforms.")
    if "prior_output_all" not in prior_capture:
        raise RuntimeError("Failed to capture prior_net output.")

    rel_trans_all = edge_capture["rel_trans_all"]
    prior_output_all = prior_capture["prior_output_all"]

    if rel_trans_all.shape[0] != edge_index_all.shape[1]:
        raise RuntimeError(
            "edge_mlp edge count does not match scene_graph.edge_index."
        )

    if prior_output_all.dim() == 3 and prior_output_all.shape[0] == 1:
        prior_output_all = prior_output_all[0]
    if prior_output_all.dim() != 2:
        raise RuntimeError(
            "Expected prior_net output [N,64], got %s"
            % (tuple(prior_output_all.shape),)
        )

    past_input = all_past_input[start:end].clone()
    past_feat_local = past_feat.detach().cpu()[start:end].clone()
    map_obs = all_map_obs[start:end].clone()
    map_feat_local = map_feat.detach().cpu()[start:end].clone()
    node_input = node_input_all[start:end].clone()
    sem = sem_all[start:end].clone()
    rel_trans = rel_trans_all[mask].clone()
    original_output = prior_output_all[start:end].clone()

    N = int(end - start)
    state_idx = _state_indices()

    payload = {
        "capture_version": "Stage-D-first-suitable-scene-v1",
        "selection_rule": (
            "first graph encountered with a target having at least one "
            "incoming edge and no incoming self-loop; first suitable target"
        ),
        "graph_id_in_batch": int(chosen["graph_id"]),
        "num_agents": N,
        "target_agent": int(target),
        "incoming_edges_to_target": int(incoming.numel()),
        "source_nodes_to_target": sources.clone(),
        "unique_source_agents": int(torch.unique(sources).numel()),
        "all_past_input": past_input,
        "all_past_feat": past_feat_local,
        "all_map_obs": map_obs,
        "all_map_feat": map_feat_local,
        "target_past_input": past_input[target].clone(),
        "target_past_feat": past_feat_local[target].clone(),
        "target_map_obs": map_obs[target].clone(),
        "target_map_feat": map_feat_local[target].clone(),
        "node_input": node_input,
        "sem": sem,
        "edge_index": local_edges.clone(),
        "rel_trans": rel_trans,
        "original_output": original_output,
        "state_indices_in_past_input": state_idx,
        "past_input_layout": (
            "4 x [state6, visibility1, vehicle_attributes2] + semantic2"
        ),
        "scope": (
            "Stage D1 capture: first suitable independent STRIVE test scene. "
            "D1 perturbs past_encoder state coordinates while map features, "
            "semantics, relative poses and topology are fixed."
        ),
    }

    torch.save(payload, output_capture)
    _DONE = True

    print()
    print("=" * 78)
    print("STAGE-D CAPTURE COMPLETE")
    print("=" * 78)
    print("selection                       : first suitable scene")
    print("agents                          : %d" % N)
    print("target agent                    : %d" % target)
    print("incoming edges to target       : %d" % int(incoming.numel()))
    print("unique source agents           : %d" % int(torch.unique(sources).numel()))
    print("all past_encoder input         : %s" % (tuple(past_input.shape),))
    print("all map observation            : %s" % (tuple(map_obs.shape),))
    print("node input                     : %s" % (tuple(node_input.shape),))
    print("state coordinates per agent    : %d" % int(state_idx.numel()))
    print("generalized D1 uncertain dims : %d" % (N * int(state_idx.numel())))
    print("wrote                          : %s" % output_capture)
    print("=" * 78)

    raise SystemExit(0)


capture_prior_stage_d = capture_stage_d
