"""One-shot Stage-B capture helper for STRIVE prior verification.

Captures, for the first node of the first prior forward pass:
  1) input to prior_net.msg[0].update_mlp
  2) output of prior_net.mlp_out

Important:
- Hooks are NOT removed from inside a running PyTorch hook. Removing them there
  can trigger "RuntimeError: OrderedDict mutated during iteration" on older
  PyTorch versions such as STRIVE's PyTorch 1.9.
- A saved-state guard makes the hooks effectively one-shot.

Usage in src/test_traffic.py, after model.eval():

    from capture_prior_stage_b import attach_stage_b_capture
    attach_stage_b_capture(model, "prior_stage_b_capture.pt")
"""
from pathlib import Path
import torch


def attach_stage_b_capture(model, output_path="prior_stage_b_capture.pt"):
    output_path = str(Path(output_path))
    state = {"saved": False}

    update_mlp = model.prior_net.msg[0].update_mlp
    output_mlp = model.prior_net.mlp_out

    def update_pre_hook(module, inputs):
        if state["saved"] or "update_input" in state:
            return
        x = inputs[0].detach().cpu()
        if x.ndim < 2 or x.shape[0] < 1:
            raise RuntimeError(f"Unexpected update_mlp input shape: {tuple(x.shape)}")
        state["update_input"] = x[0].clone()
        state["full_update_input_shape"] = tuple(x.shape)

    def output_hook(module, inputs, output):
        if state["saved"] or "update_input" not in state:
            return

        y = output.detach().cpu()
        if y.ndim < 2 or y.shape[0] < 1:
            raise RuntimeError(f"Unexpected mlp_out output shape: {tuple(y.shape)}")

        payload = {
            "update_input": state["update_input"],
            "original_output": y[0].clone(),
            "full_update_input_shape": state["full_update_input_shape"],
            "full_output_shape": tuple(y.shape),
            "scope": "prior_net.msg[0].update_mlp -> prior_net.mlp_out, first node",
        }

        torch.save(payload, output_path)
        state["saved"] = True

        print(f"Saved STRIVE Stage-B capture: {output_path}")
        print("  update_input:", tuple(payload["update_input"].shape))
        print("  original_output:", tuple(payload["original_output"].shape))

    update_handle = update_mlp.register_forward_pre_hook(update_pre_hook)
    output_handle = output_mlp.register_forward_hook(output_hook)

    print("Attached STRIVE Stage-B one-shot capture hooks.")

    # Return handles in case caller wants to remove them later, but do not
    # remove them from inside the hook itself.
    return update_handle, output_handle
