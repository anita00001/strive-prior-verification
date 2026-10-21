"""Capture one real fixed STRIVE prior graph for Stage C verification.

Stage C captures the actual input to model.prior_net for one graph that has
at least one interaction edge. It also captures the exact relative-pose
features that STRIVE feeds to edge_mlp, avoiding any reimplementation of
transform2frame during verification.

Add after model.eval() in src/test_traffic.py:

    from capture_prior_stage_c import attach_stage_c_capture
    attach_stage_c_capture(model, "prior_stage_c_capture.pt")

The hooks are guarded but deliberately not removed from inside a hook. This
avoids "OrderedDict mutated during iteration" on STRIVE's PyTorch 1.9.
"""
from pathlib import Path
import torch


def attach_stage_c_capture(model, output_path="prior_stage_c_capture.pt"):
    output_path = str(Path(output_path))
    state = {"saved": False}

    prior_net = model.prior_net
    edge_mlp = model.prior_net.msg[0].edge_mlp

    def prior_pre_hook(module, inputs):
        if state["saved"] or "node_input" in state:
            return

        scene_graph = inputs[0]
        x = scene_graph.x.detach().cpu()
        sem = scene_graph.sem.detach().cpu()
        edge_index = scene_graph.edge_index.detach().cpu().long()

        # Prefer PyG Batch.ptr so that we capture one actual scene rather than
        # a whole minibatch of disconnected scenes.
        if hasattr(scene_graph, "ptr") and scene_graph.ptr is not None:
            ptr = scene_graph.ptr.detach().cpu().long()
            candidates = [(int(ptr[g]), int(ptr[g + 1]))
                          for g in range(len(ptr) - 1)]
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
            if bool(mask.any()):
                local_edges = edge_index[:, mask] - start
                # Pick an agent that receives at least one message so the
                # aggregation path is definitely exercised.
                target_agent = int(local_edges[1, 0])
                chosen = (
                    graph_id, start, end, mask, local_edges, target_agent
                )
                break

        if chosen is None:
            # Wait for a later prior forward pass containing a graph with edges.
            return

        graph_id, start, end, mask, local_edges, target_agent = chosen

        state["graph_id"] = graph_id
        state["start"] = start
        state["end"] = end
        state["edge_mask"] = mask
        state["node_input"] = x[start:end].clone()
        state["sem"] = sem[start:end].clone()
        state["edge_index"] = local_edges.clone()
        state["target_agent"] = target_agent

        print(
            f"Stage-C candidate: graph={graph_id}, "
            f"nodes={end-start}, edges={local_edges.shape[1]}, "
            f"target_agent={target_agent}"
        )

    def edge_pre_hook(module, inputs):
        if state["saved"] or "node_input" not in state or "rel_trans" in state:
            return

        msg_in = inputs[0].detach().cpu()
        mask = state["edge_mask"]

        if msg_in.ndim != 2 or msg_in.shape[-1] < 4:
            raise RuntimeError(
                f"Unexpected edge_mlp input shape: {tuple(msg_in.shape)}"
            )
        if msg_in.shape[0] != mask.numel():
            raise RuntimeError(
                "edge_mlp edge count does not match captured edge_index."
            )

        # In STRIVE's non-GRU prior, the final 4 edge-MLP input dimensions
        # are exactly rel_trans=(x,y,hx,hy) of source j in target i's frame.
        state["rel_trans"] = msg_in[mask, -4:].clone()

    def prior_post_hook(module, inputs, output):
        if (
            state["saved"]
            or "node_input" not in state
            or "rel_trans" not in state
        ):
            return

        out = output.detach().cpu()
        start, end = state["start"], state["end"]

        payload = {
            "node_input": state["node_input"],
            "sem": state["sem"],
            "edge_index": state["edge_index"],
            "rel_trans": state["rel_trans"],
            "original_output": out[start:end].clone(),
            "target_agent": state["target_agent"],
            "graph_id_in_batch": state["graph_id"],
            "scope": (
                "fixed-scene full prior_net: mlp_in -> edge_mlp -> "
                "max aggregation -> update_mlp -> mlp_out"
            ),
        }

        torch.save(payload, output_path)
        state["saved"] = True

        print(f"Saved STRIVE Stage-C capture: {output_path}")
        print("  node_input     :", tuple(payload["node_input"].shape))
        print("  sem            :", tuple(payload["sem"].shape))
        print("  edge_index     :", tuple(payload["edge_index"].shape))
        print("  rel_trans      :", tuple(payload["rel_trans"].shape))
        print("  original_output:", tuple(payload["original_output"].shape))
        print("  target_agent   :", payload["target_agent"])

    prior_net.register_forward_pre_hook(prior_pre_hook)
    edge_mlp.register_forward_pre_hook(edge_pre_hook)
    prior_net.register_forward_hook(prior_post_hook)

    print("Attached STRIVE Stage-C one-shot capture hooks.")
