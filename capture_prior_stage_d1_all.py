#!/usr/bin/env python3
"""
Capture Stage D1-generalized inputs for the SAME scene used by prior_stage_c_capture.pt.

This helper is intended to be imported from STRIVE/src/test_traffic.py immediately
after:

    pred = model(scene_graph, map_idx, map_env, use_post_mean=True)

It searches the test stream for the exact C-stage scene by comparing the reconstructed
prior input [past_feat, map_feat, sem] against reference_capture["node_input"].

When a match is found it records, for ALL agents:
  * exact 38-D input to model.past_encoder
  * past features (64-D)
  * map features (64-D)
  * normalized scene_graph.past [N,4,6]
  * past visibility, vehicle attributes, semantics, edge_index
  * target agent and prior output

The 38-D MLP past input has layout:
    4 x [state6, visibility1, vehicle_attributes2] + semantic2
so the 24 state coordinates are:
    0:6, 9:15, 18:24, 27:33.

This capture does NOT itself perform verification.
Compatible with the older STRIVE/PyTorch environment (no weights_only=True).
"""

from __future__ import print_function
import os
import torch

_REF_CACHE = None
_DONE = False


def _squeeze_optional_batch(x):
    if torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] == 1:
        # Only squeeze when doing so makes sense for a stored node tensor.
        if x.dim() >= 3:
            return x[0]
    return x


def _reference_node_input(ref):
    if "node_input" not in ref:
        raise KeyError(
            "Reference C capture has no 'node_input'. "
            "This helper needs prior_stage_c_capture.pt."
        )
    x = ref["node_input"]
    if not torch.is_tensor(x):
        raise TypeError("reference 'node_input' is not a tensor")
    x = x.detach().cpu()
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() != 2:
        raise RuntimeError(
            "Expected reference node_input to be [N,130] or [1,N,130], got %s"
            % (tuple(x.shape),)
        )
    return x


def _reference_output_for_target(ref, target):
    if "original_output" not in ref:
        return None
    y = ref["original_output"]
    if not torch.is_tensor(y):
        return None
    y = y.detach().cpu()
    if y.dim() == 3 and y.shape[0] == 1:
        y = y[0]
    if y.dim() == 2:
        if target < y.shape[0]:
            return y[target].reshape(-1)
        if y.shape[0] == 1:
            return y[0].reshape(-1)
    if y.dim() == 1:
        return y.reshape(-1)
    return None


def capture_d1_all_if_match(
    model,
    scene_graph,
    map_idx,
    map_env,
    reference_capture="./prior_stage_c_capture.pt",
    output_capture="./prior_stage_d1_all_capture.pt",
    match_atol=2e-5,
):
    global _REF_CACHE, _DONE

    if _DONE:
        return False

    if os.path.exists(output_capture):
        print(
            "[D1-all capture] Output already exists: %s" % output_capture
        )
        print("[D1-all capture] Remove it if you intentionally want to recapture.")
        raise SystemExit(0)

    if not os.path.exists(reference_capture):
        raise RuntimeError(
            "[D1-all capture] Reference C capture not found: %s"
            % reference_capture
        )

    if _REF_CACHE is None:
        _REF_CACHE = torch.load(reference_capture, map_location="cpu")

    ref = _REF_CACHE
    ref_node = _reference_node_input(ref)

    captured = {}

    def pre_hook(module, inputs):
        if len(inputs) < 1:
            raise RuntimeError("past_encoder pre-hook received no input")
        captured["past_input"] = inputs[0].detach().cpu().clone()

    handle = model.past_encoder.register_forward_pre_hook(pre_hook)

    try:
        with torch.no_grad():
            # STRIVE prior uses the final past pose for map extraction / graph position.
            scene_graph.pos = scene_graph.past[:, -1, :4]

            map_feat = model.encode_map(scene_graph, map_idx, map_env)
            past_feat = model.encode_past(scene_graph)

            if "past_input" not in captured:
                raise RuntimeError("Failed to intercept model.past_encoder input")

            cur_node = torch.cat(
                [past_feat, map_feat, scene_graph.sem],
                dim=-1,
            ).detach().cpu()

    finally:
        handle.remove()

    if tuple(cur_node.shape) != tuple(ref_node.shape):
        return False

    match_err = float((cur_node - ref_node).abs().max())
    if match_err > float(match_atol):
        return False

    target = int(ref["target_agent"])

    # Re-evaluate the prior only after the scene has matched.
    with torch.no_grad():
        scene_graph.pos = scene_graph.past[:, -1, :4]
        mu, var = model.prior(scene_graph, map_feat, past_feat)
        logvar = torch.log(var)
        prior_out = torch.cat([mu, logvar], dim=-1).detach().cpu()

    ref_target_out = _reference_output_for_target(ref, target)
    prior_target_err = None
    if ref_target_out is not None and ref_target_out.numel() == 64:
        prior_target_err = float(
            (prior_out[target].reshape(-1) - ref_target_out).abs().max()
        )

    past_input = captured["past_input"]
    if past_input.dim() != 2 or past_input.shape[1] != 38:
        raise RuntimeError(
            "Expected past_encoder input [N,38], got %s"
            % (tuple(past_input.shape),)
        )

    # state6 within each 9-D per-time-step block.
    state_indices = []
    for t in range(4):
        start = t * 9
        state_indices.extend(range(start, start + 6))
    state_indices = torch.tensor(state_indices, dtype=torch.long)

    save = {
        "capture_version": "D1-generalized-all-agents-v1",
        "reference_c_capture": str(reference_capture),
        "target_agent": target,
        "all_past_input": past_input,
        "all_past_feat": past_feat.detach().cpu().clone(),
        "all_map_feat": map_feat.detach().cpu().clone(),
        "scene_past_normalized": scene_graph.past.detach().cpu().clone(),
        "past_vis": scene_graph.past_vis.detach().cpu().clone(),
        "lw": scene_graph.lw.detach().cpu().clone(),
        "sem": scene_graph.sem.detach().cpu().clone(),
        "edge_index": scene_graph.edge_index.detach().cpu().clone(),
        "prior_output_all": prior_out,
        "state_indices_in_past_input": state_indices,
        "past_input_layout": (
            "4 x [state6, visibility1, vehicle_attributes2] + semantic2"
        ),
        "reference_node_input_max_abs_error": match_err,
        "reference_target_output_max_abs_error": prior_target_err,
    }

    torch.save(save, output_capture)
    _DONE = True

    print()
    print("=" * 76)
    print("STAGE D1-GENERALIZED CAPTURE COMPLETE")
    print("=" * 76)
    print("matched C-stage scene          : YES")
    print("agents                         : %d" % past_input.shape[0])
    print("past_encoder input             : %s" % (tuple(past_input.shape),))
    print("state coordinates per agent    : %d" % state_indices.numel())
    print(
        "total candidate state coords : %d"
        % (past_input.shape[0] * state_indices.numel())
    )
    print("C node-input match error       : %.8g" % match_err)
    if prior_target_err is not None:
        print("C target-output match error    : %.8g" % prior_target_err)
    print("wrote                          : %s" % output_capture)
    print("=" * 76)
    print("Stopping test_traffic.py because the requested capture is complete.")
    raise SystemExit(0)
