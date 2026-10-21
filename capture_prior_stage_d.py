"""Stage D capture helper for STRIVE full conditional-prior verification.

Captures one real target agent with at least one incoming prior-GNN edge.

Captured target-side inputs:
  - exact input to past_encoder (MLP configuration)
  - exact local map raster entering map_conv
  - semantic class
  - fixed-scene prior graph / relative edge features
  - original prior_net output
  - original past/map feature vectors for equivalence checks

Add after model.eval() in src/test_traffic.py:

    from capture_prior_stage_d import attach_stage_d_capture
    attach_stage_d_capture(model, "prior_stage_d_capture.pt")

The hooks are guarded but are not removed from inside a hook, which avoids the
PyTorch-1.9 "OrderedDict mutated during iteration" problem.
"""

from pathlib import Path
import torch


def attach_stage_d_capture(model, output_path="prior_stage_d_capture.pt"):
    if getattr(model, "traj_encoder_type", None) != "mlp":
        raise RuntimeError(
            "This Stage-D helper currently targets STRIVE's MLP past encoder. "
            f"Found traj_encoder_type={getattr(model, 'traj_encoder_type', None)!r}."
        )

    output_path = str(Path(output_path))
    state = {"saved": False}

    prior_net = model.prior_net
    edge_mlp = model.prior_net.msg[0].edge_mlp

    def map_pre_hook(module, inputs):
        if state["saved"] or "map_obs_all" in state:
            return
        x = inputs[0].detach().cpu()
        if x.ndim != 4:
            raise RuntimeError(f"Unexpected map_conv input: {tuple(x.shape)}")
        state["map_obs_all"] = x.clone()

    def past_pre_hook(module, inputs):
        if state["saved"] or "past_in_all" in state:
            return
        x = inputs[0].detach().cpu()
        if x.ndim != 2:
            raise RuntimeError(f"Unexpected past_encoder input: {tuple(x.shape)}")
        state["past_in_all"] = x.clone()

    def prior_pre_hook(module, inputs):
        if state["saved"] or "node_input" in state:
            return

        if "map_obs_all" not in state or "past_in_all" not in state:
            # Wait until the forward pass where both upstream branches were seen.
            return

        scene_graph = inputs[0]
        x = scene_graph.x.detach().cpu()
        sem = scene_graph.sem.detach().cpu()
        edge_index = scene_graph.edge_index.detach().cpu().long()

        if state["map_obs_all"].shape[0] != x.shape[0]:
            raise RuntimeError(
                "map_conv input node count does not match prior_net input."
            )
        if state["past_in_all"].shape[0] != x.shape[0]:
            raise RuntimeError(
                "past_encoder input node count does not match prior_net input."
            )

        if hasattr(scene_graph, "ptr") and scene_graph.ptr is not None:
            ptr = scene_graph.ptr.detach().cpu().long()
            candidates = [
                (int(ptr[g]), int(ptr[g + 1]))
                for g in range(len(ptr) - 1)
            ]
        else:
            candidates = [(0, int(x.shape[0]))]

        src_all = edge_index[0]
        dst_all = edge_index[1]

        chosen = None
        for graph_id, (start, end) in enumerate(candidates):
            mask = (
                (src_all >= start) & (src_all < end)
                & (dst_all >= start) & (dst_all < end)
            )
            if not bool(mask.any()):
                continue

            local_edges = edge_index[:, mask] - start

            # Prefer a target with at least one incoming edge and no self-loop.
            for eid in range(local_edges.shape[1]):
                target = int(local_edges[1, eid])
                incoming = torch.nonzero(
                    local_edges[1] == target, as_tuple=False
                ).flatten()
                sources = local_edges[0, incoming]
                if incoming.numel() > 0 and not bool((sources == target).any()):
                    chosen = (
                        graph_id, start, end, mask, local_edges, target
                    )
                    break
            if chosen is not None:
                break

        if chosen is None:
            return

        graph_id, start, end, mask, local_edges, target = chosen
        global_target = start + target

        # prior_net input layout = [past_feat(64), map_feat(64), sem(2)]
        if x.shape[1] != 130:
            raise RuntimeError(
                f"Expected prior_net input dim 130, got {x.shape[1]}"
            )

        state.update(
            {
                "graph_id": graph_id,
                "start": start,
                "end": end,
                "edge_mask": mask,
                "node_input": x[start:end].clone(),
                "sem": sem[start:end].clone(),
                "edge_index": local_edges.clone(),
                "target_agent": target,
                "target_global": global_target,
                "target_past_input": state["past_in_all"][global_target].clone(),
                "target_map_obs": state["map_obs_all"][global_target].clone(),
                "target_past_feat": x[global_target, :64].clone(),
                "target_map_feat": x[global_target, 64:128].clone(),
            }
        )

        print(
            f"Stage-D candidate: graph={graph_id}, nodes={end-start}, "
            f"edges={local_edges.shape[1]}, target_agent={target}"
        )

    def edge_pre_hook(module, inputs):
        if state["saved"] or "node_input" not in state or "rel_trans" in state:
            return

        msg_in = inputs[0].detach().cpu()
        mask = state["edge_mask"]

        if msg_in.ndim != 2 or msg_in.shape[-1] < 4:
            raise RuntimeError(
                f"Unexpected prior edge_mlp input shape: {tuple(msg_in.shape)}"
            )
        if msg_in.shape[0] != mask.numel():
            raise RuntimeError(
                "edge_mlp edge count does not match captured edge_index."
            )

        state["rel_trans"] = msg_in[mask, -4:].clone()

    def prior_post_hook(module, inputs, output):
        required = {
            "node_input", "rel_trans", "target_past_input",
            "target_map_obs", "target_past_feat", "target_map_feat"
        }
        if state["saved"] or not required.issubset(state):
            return

        out = output.detach().cpu()
        start, end = state["start"], state["end"]

        payload = {
            "target_past_input": state["target_past_input"],
            "target_map_obs": state["target_map_obs"],
            "target_past_feat": state["target_past_feat"],
            "target_map_feat": state["target_map_feat"],
            "node_input": state["node_input"],
            "sem": state["sem"],
            "edge_index": state["edge_index"],
            "rel_trans": state["rel_trans"],
            "original_output": out[start:end].clone(),
            "target_agent": state["target_agent"],
            "graph_id_in_batch": state["graph_id"],
            "scope": (
                "Stage D raw target branches + fixed-scene prior GNN: "
                "past_encoder + map_conv/GroupNorm/map_feature + "
                "prior_net message passing + prior output"
            ),
        }

        torch.save(payload, output_path)
        state["saved"] = True

        print(f"Saved STRIVE Stage-D capture: {output_path}")
        for key in [
            "target_past_input", "target_map_obs",
            "target_past_feat", "target_map_feat",
            "node_input", "sem", "edge_index",
            "rel_trans", "original_output"
        ]:
            value = payload[key]
            print(f"  {key:20s}: {tuple(value.shape)}")
        print("  target_agent        :", payload["target_agent"])

    model.map_conv.register_forward_pre_hook(map_pre_hook)
    model.past_encoder.register_forward_pre_hook(past_pre_hook)
    prior_net.register_forward_pre_hook(prior_pre_hook)
    edge_mlp.register_forward_pre_hook(edge_pre_hook)
    prior_net.register_forward_hook(prior_post_hook)

    print("Attached STRIVE Stage-D one-shot capture hooks.")
