#!/usr/bin/env python3
"""
STRIVE Stage D2-generalized patch custom-normalization backward-CROWN verifier.

This verifier is model-specific.  It keeps ordinary CROWN-style affine bounds
through every ReLU/Linear/Conv layer, but treats GroupNorm(1,C) and LayerNorm
as ONE custom operator instead of decomposing them into mean/square/sqrt/
reciprocal/multiply nodes.

That is the key difference from the earlier failed formulations.

Scope
-----
Current captured scene:
  * one selected target agent;
  * one incoming message edge;
  * all four channels in one square map patch are uncertain jointly;
  * everything else fixed.

Property
--------
For every output y_i,

    ||x-x0||_inf <= epsilon
        => |y_i(x)-y_i(x0)| <= delta.

Custom normalization relaxation
-------------------------------
For a normalization group

    y_i = gamma_i (x_i - mean(x)) / s + beta_i
    s = sqrt(Var(x) + eps_norm),

and an arbitrary backward affine coefficient vector a,

    a^T y = beta_term + z / s
    z = q^T x
    q = a*gamma - mean(a*gamma).

Given certified
    z in [zL,zU],
    s in [sL,sU],
the verifier eliminates s with a sound piecewise ratio envelope and returns
a LINEAR bound in x.  Therefore the affine dependence is preserved backward.

For LayerNorm variance lower bounds, the verifier uses THREE sound lemmas:
  1) exact minimum variance over the current elementwise box (1-D convex solve);
  2) a CROWN pairwise-difference invariant:
         Var(x) = 1/n^2 sum_{i<j} (x_i-x_j)^2.
     Every pair difference is itself bounded by backward CROWN through the
     current prefix.  Summing any certified pairwise square lower bounds gives
     a sound variance invariant.

For GroupNorm in the map CNN, tight value/derivative bounds from the existing
K-root map sensitivity propagation are reused to obtain positive variance
invariants.  Those were already observed to remain healthy through GN6.

Architecture used for the selected one-edge scene
-------------------------------------------------
map patch (K = 4 x patch_size x patch_size scalars)
 -> six Conv/GN/ReLU blocks
 -> 64-D map feature
 -> prior_mlp_in
 -> edge_mlp (single incoming edge; max is identity)
 -> update_mlp
 -> prior_mlp_out
 -> 64 outputs

The prior is represented as a sequential CROWN graph by retaining h while the
edge MLP is evaluated, i.e. a state [h, message].  This preserves the branch
correlation exactly at the affine-graph level.

Verdicts
--------
CERTIFIED:
    all 64 output intervals are within nominal +/- delta on every terminal box.

NOT_PROVED:
    complete finite sound bounds were obtained, but at least one output exceeds
    delta.

NO_VERIFICATION_RESULT:
    a numerical/tooling/model-assumption check failed on at least one terminal
    region.

Notes
-----
* This is backward CROWN / DeepPoly-style linear relaxation with a custom
  normalization relaxation.  It is NOT stock alpha-beta-CROWN branch-and-bound
  and does not claim alpha optimization.
* Computation is float64 with explicit outward slack for interval quantities.
"""

import argparse
import csv
import heapq
import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from verify_prior_stage_d1 import build_model
from verify_prior_stage_d2_compositional_lipschitz import (
    conv_value_interval,
    conv_deriv_interval,
    groupnorm1_sensitivity,
    relu_propagate,
    linear_value_interval as map_linear_value_interval,
    linear_deriv_interval as map_linear_deriv_interval,
    mvt_value_box,
    max_abs_interval,
    outward,
)


# ============================================================================
# Basic interval helpers
# ============================================================================

def box_concretize(A, bias, lower, upper, side):
    """Concretize affine rows A x + bias over a box."""
    A_pos = torch.clamp(A, min=0.0)
    A_neg = torch.clamp(A, max=0.0)

    if side == "lower":
        return (
            (A_pos * lower + A_neg * upper).reshape(A.shape[0], -1).sum(dim=1)
            + bias
        )
    if side == "upper":
        return (
            (A_pos * upper + A_neg * lower).reshape(A.shape[0], -1).sum(dim=1)
            + bias
        )
    raise ValueError(side)


def center_radius_concretize(A, bias, center, radius, side):
    nominal = (A * center).reshape(A.shape[0], -1).sum(dim=1) + bias
    rad = (A.abs() * radius).reshape(A.shape[0], -1).sum(dim=1)
    return nominal - rad if side == "lower" else nominal + rad


def exact_min_variance_box_1d(lower, upper, iterations=100):
    """Sound min Var(x) over an elementwise box [lower,upper].

    For n coordinates:
        n * min Var = min_m sum_i dist(m,[l_i,u_i])^2.

    The objective is convex in scalar m.  Bisection solves its monotone
    derivative.  Returned value is slightly reduced for numerical safety.
    """
    l = lower.reshape(-1).double()
    u = upper.reshape(-1).double()

    if l.numel() == 0:
        raise RuntimeError("empty normalization group")

    lo = float(l.min())
    hi = float(u.max())

    for _ in range(iterations):
        m = 0.5 * (lo + hi)
        mt = torch.tensor(m, dtype=l.dtype, device=l.device)

        grad = (
            torch.where(mt < l, mt - l, torch.zeros_like(l))
            + torch.where(mt > u, mt - u, torch.zeros_like(u))
        ).sum()

        if float(grad) < 0.0:
            lo = m
        else:
            hi = m

    m = torch.tensor(
        0.5 * (lo + hi),
        dtype=l.dtype,
        device=l.device,
    )
    dist = torch.where(
        m < l,
        l - m,
        torch.where(m > u, m - u, torch.zeros_like(l)),
    )
    var = (dist * dist).mean()

    # Tiny downward pad keeps the lower bound conservative under floating point.
    return torch.clamp(
        var - (1e-14 + 1e-12 * var.abs()),
        min=0.0,
    )


def variance_upper_from_box(lower, upper):
    mean_l = lower.mean()
    mean_u = upper.mean()

    centered_l = lower - mean_u
    centered_u = upper - mean_l

    sq_max = torch.maximum(
        centered_l * centered_l,
        centered_u * centered_u,
    )
    var_u = sq_max.mean()
    return var_u + (1e-14 + 1e-12 * var_u.abs())


def ratio_envelope(A, x_l, x_u, gamma, beta, var_l, var_u, eps_norm, side):
    """Custom backward affine relaxation for one normalization group.

    A: [S,N] coefficients on normalized output.
    Returns coefficients [S,N] on normalization input and bias [S].
    """
    A = A.double()
    x_l = x_l.reshape(-1).double()
    x_u = x_u.reshape(-1).double()
    gamma = gamma.reshape(-1).double()
    beta = beta.reshape(-1).double()

    if A.shape[1] != x_l.numel():
        raise RuntimeError(
            f"normalization shape mismatch: A={A.shape}, group={x_l.numel()}"
        )

    # a^T gamma * (x-mean)/s = q^T x / s.
    p = A * gamma.unsqueeze(0)
    q = p - p.mean(dim=1, keepdim=True)

    q_pos = torch.clamp(q, min=0.0)
    q_neg = torch.clamp(q, max=0.0)

    z_l = (q_pos * x_l + q_neg * x_u).sum(dim=1)
    z_u = (q_pos * x_u + q_neg * x_l).sum(dim=1)

    var_l = max(float(var_l), 0.0)
    var_u = max(float(var_u), var_l)

    s_l = math.sqrt(var_l + float(eps_norm))
    s_u = math.sqrt(var_u + float(eps_norm))

    if not (math.isfinite(s_l) and math.isfinite(s_u) and s_l > 0.0):
        raise RuntimeError(
            f"invalid certified normalization scale interval [{s_l},{s_u}]"
        )

    # t = 1/s.
    t_l = 1.0 / s_u
    t_u = 1.0 / s_l

    alpha = torch.empty_like(z_l)
    intercept = torch.zeros_like(z_l)

    neg = z_u <= 0
    pos = z_l >= 0
    cross = ~(neg | pos)

    if side == "upper":
        # max_s z/s:
        #   negative z -> t_l*z
        #   positive z -> t_u*z
        alpha[neg] = t_l
        alpha[pos] = t_u

        if bool(cross.any()):
            zl = z_l[cross]
            zu = z_u[cross]
            den = zu - zl
            au = (t_u * zu - t_l * zl) / den
            bu = t_l * zl - au * zl
            alpha[cross] = au
            intercept[cross] = bu

    elif side == "lower":
        # min_s z/s:
        #   negative z -> t_u*z
        #   positive z -> t_l*z
        alpha[neg] = t_u
        alpha[pos] = t_l

        if bool(cross.any()):
            zl = z_l[cross]
            zu = z_u[cross]
            den = zu - zl
            al = (t_l * zu - t_u * zl) / den
            bl = t_u * zl - al * zl
            alpha[cross] = al
            intercept[cross] = bl
    else:
        raise ValueError(side)

    A_in = alpha.unsqueeze(1) * q
    bias = (A * beta.unsqueeze(0)).sum(dim=1) + intercept

    return A_in, bias


# ============================================================================
# Prior-vector CROWN operations
# ============================================================================

class VecOp:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, A, side):
        raise NotImplementedError


class AffineVecOp(VecOp):
    def __init__(self, W, b, name):
        self.W = W.detach().double().clone()
        self.b = b.detach().double().clone()
        self.name = name

    def forward(self, x):
        return x @ self.W.t() + self.b

    def backward(self, A, side):
        return A @ self.W, A @ self.b


class ReLUVecOp(VecOp):
    def __init__(self, lower, upper, start, end, name):
        self.lower = lower.detach().double().clone()
        self.upper = upper.detach().double().clone()
        self.start = int(start)
        self.end = int(end)
        self.name = name

    def forward(self, x):
        y = x.clone()
        y[self.start:self.end] = torch.relu(y[self.start:self.end])
        return y

    def _slopes(self):
        l = self.lower[self.start:self.end]
        u = self.upper[self.start:self.end]

        pos = l >= 0
        neg = u <= 0
        unstable = ~(pos | neg)

        alpha_u = torch.zeros_like(l)
        beta_u = torch.zeros_like(l)
        alpha_l = torch.zeros_like(l)
        beta_l = torch.zeros_like(l)

        alpha_u[pos] = 1.0
        alpha_l[pos] = 1.0

        if bool(unstable.any()):
            lu = l[unstable]
            uu = u[unstable]
            au = uu / torch.clamp(uu - lu, min=1e-30)
            bu = -lu * au
            alpha_u[unstable] = au
            beta_u[unstable] = bu

            # Standard adaptive CROWN lower line: either y>=x or y>=0.
            use_identity = uu >= -lu
            alpha_l[unstable] = use_identity.to(l.dtype)

        return alpha_l, beta_l, alpha_u, beta_u

    def backward(self, A, side):
        A = A.double()
        out = A.clone()

        alpha_l, beta_l, alpha_u, beta_u = self._slopes()
        As = A[:, self.start:self.end]

        if side == "lower":
            choose_upper = As < 0
        elif side == "upper":
            choose_upper = As >= 0
        else:
            raise ValueError(side)

        slope = torch.where(
            choose_upper,
            alpha_u.unsqueeze(0),
            alpha_l.unsqueeze(0),
        )
        intercept = torch.where(
            choose_upper,
            beta_u.unsqueeze(0),
            beta_l.unsqueeze(0),
        )

        out[:, self.start:self.end] = As * slope
        bias = (As * intercept).sum(dim=1)
        return out, bias


class NormVecOp(VecOp):
    def __init__(
        self,
        lower,
        upper,
        gamma,
        beta,
        eps_norm,
        var_lower,
        var_upper,
        start,
        end,
        name,
        variance_meta=None,
    ):
        self.lower = lower.detach().double().clone()
        self.upper = upper.detach().double().clone()
        self.gamma = gamma.detach().double().clone()
        self.beta = beta.detach().double().clone()
        self.eps_norm = float(eps_norm)
        self.var_lower = float(var_lower)
        self.var_upper = float(var_upper)
        self.start = int(start)
        self.end = int(end)
        self.name = name
        self.variance_meta = variance_meta or {}

    def forward(self, x):
        y = x.clone()
        xs = y[self.start:self.end]
        mean = xs.mean()
        var = ((xs - mean) ** 2).mean()
        norm = (xs - mean) / torch.sqrt(var + self.eps_norm)
        y[self.start:self.end] = norm * self.gamma + self.beta
        return y

    def backward(self, A, side):
        out = A.double().clone()
        As = A[:, self.start:self.end]

        Ain, bias = ratio_envelope(
            As,
            self.lower[self.start:self.end],
            self.upper[self.start:self.end],
            self.gamma,
            self.beta,
            self.var_lower,
            self.var_upper,
            self.eps_norm,
            side,
        )
        out[:, self.start:self.end] = Ain
        return out, bias


def crown_affine_forms(ops, C):
    """Return lower/upper affine forms wrt the sequence root."""
    A_l = C.double().clone()
    A_u = C.double().clone()
    b_l = torch.zeros(C.shape[0], dtype=torch.float64)
    b_u = torch.zeros_like(b_l)

    for op in reversed(ops):
        A_l, db_l = op.backward(A_l, "lower")
        A_u, db_u = op.backward(A_u, "upper")
        b_l = b_l + db_l
        b_u = b_u + db_u

    return A_l, b_l, A_u, b_u


def crown_specs(ops, root_l, root_u, C, spec_batch=128):
    """Bound C @ current_state for arbitrary C rows."""
    lowers = []
    uppers = []

    for s in range(0, C.shape[0], spec_batch):
        c = C[s:s + spec_batch].double()
        A_l, b_l, A_u, b_u = crown_affine_forms(ops, c)
        lowers.append(box_concretize(A_l, b_l, root_l, root_u, "lower"))
        uppers.append(box_concretize(A_u, b_u, root_l, root_u, "upper"))

    return torch.cat(lowers), torch.cat(uppers)


def crown_state_bounds(ops, root_l, root_u, dim, spec_batch=128):
    eye = torch.eye(dim, dtype=torch.float64)
    return crown_specs(ops, root_l, root_u, eye, spec_batch)


def pairwise_variance_crown_lower(
    ops,
    root_l,
    root_u,
    state_dim,
    start,
    end,
    pair_batch=512,
):
    """Sound LayerNorm variance lower bound using all pair differences."""
    n = end - start
    if n <= 1:
        return 0.0

    total_sq_lb = torch.tensor(0.0, dtype=torch.float64)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row = torch.zeros(state_dim, dtype=torch.float64)
            row[start + i] = 1.0
            row[start + j] = -1.0
            rows.append(row)

            if len(rows) >= pair_batch:
                C = torch.stack(rows)
                l, u = crown_specs(
                    ops, root_l, root_u, C, spec_batch=pair_batch
                )
                minabs = torch.where(
                    l > 0,
                    l,
                    torch.where(u < 0, -u, torch.zeros_like(l)),
                )
                total_sq_lb = total_sq_lb + (minabs * minabs).sum()
                rows = []

    if rows:
        C = torch.stack(rows)
        l, u = crown_specs(
            ops, root_l, root_u, C, spec_batch=pair_batch
        )
        minabs = torch.where(
            l > 0,
            l,
            torch.where(u < 0, -u, torch.zeros_like(l)),
        )
        total_sq_lb = total_sq_lb + (minabs * minabs).sum()

    return float(total_sq_lb / float(n * n))


def nominal_projection_variance_crown_lower(
    ops,
    root_l,
    root_u,
    nominal_state,
    start,
    end,
):
    """Sound variance lower bound from a nominal-aligned CROWN projection.

    Let z be the pre-normalization vector (length n), z0 its nominal value,
    and c0 = z0 - mean(z0).  For c0 != 0 choose

        w = c0 / ||c0||_2.

    Since w is unit length and zero mean,

        Var(z) = ||Pz||_2^2 / n >= (w^T z)^2 / n.

    We CROWN-bound w^T z through the current prefix.  If the interval excludes
    zero, this gives a positive variance invariant.  At the nominal point this
    single projection exactly recovers the nominal variance.
    """
    z0 = nominal_state[start:end].double()
    n = int(z0.numel())

    if n <= 1:
        return 0.0, 0.0, 0.0

    centered = z0 - z0.mean()
    norm = torch.linalg.vector_norm(centered)

    if float(norm) <= 1e-30:
        return 0.0, 0.0, 0.0

    w = centered / norm

    C = torch.zeros(
        1,
        nominal_state.numel(),
        dtype=torch.float64,
    )
    C[0, start:end] = w

    l, u = crown_specs(
        ops,
        root_l,
        root_u,
        C,
        spec_batch=1,
    )

    lo = float(l[0])
    hi = float(u[0])

    if lo > 0.0:
        minabs = lo
    elif hi < 0.0:
        minabs = -hi
    else:
        minabs = 0.0

    var_lb = (minabs * minabs) / float(n)

    # Conservative downward numerical pad.
    var_lb = max(
        0.0,
        var_lb - (1e-14 + 1e-12 * abs(var_lb)),
    )

    return var_lb, lo, hi


# ============================================================================
# Build a sequential one-edge prior graph rooted at 64-D map feature
# ============================================================================

def _layernorm_params(layer):
    if not isinstance(layer, nn.LayerNorm):
        raise TypeError(type(layer))
    if len(layer.normalized_shape) != 1:
        raise NotImplementedError(
            f"Only 1-D LayerNorm is supported, got {layer.normalized_shape}"
        )

    n = int(layer.normalized_shape[0])
    if layer.elementwise_affine:
        gamma = layer.weight.detach().double()
        beta = layer.bias.detach().double()
    else:
        gamma = torch.ones(n, dtype=torch.float64)
        beta = torch.zeros(n, dtype=torch.float64)
    return gamma, beta, float(layer.eps)


def _fold_linear_with_constants(layer, variable_indices, constant_vector):
    W = layer.weight.detach().double()
    b = (
        layer.bias.detach().double().clone()
        if layer.bias is not None
        else torch.zeros(W.shape[0], dtype=torch.float64)
    )

    variable_indices = torch.tensor(variable_indices, dtype=torch.long)
    mask = torch.ones(W.shape[1], dtype=torch.bool)
    mask[variable_indices] = False
    const_idx = torch.arange(W.shape[1])[mask]

    Wvar = W[:, variable_indices]

    if const_idx.numel():
        c = constant_vector[const_idx].double()
        b = b + W[:, const_idx] @ c

    return Wvar, b


def combined_crown_specs_direct(
    prior_ops,
    map_ctx,
    C,
    map_spec_batch=8,
):
    """Bound C @ current_prior_state directly to the original map-patch roots.

    This is the key v3 change.  v1/v2 bounded prior intermediates relative to
    an independent 64-D map-feature box and only mapped the FINAL output back
    through the map CNN.  That destroyed the map-feature correlation before
    the LayerNorm variance invariants were computed.

    Here every prior intermediate specification is propagated:
        current prior state
          -> backward through prior_ops
          -> 64 map features
          -> backward through custom map-CROWN ops
          -> original map raster
          -> concretize only the uncertain map-patch values.

    Therefore all intermediate ReLU bounds and LayerNorm variance invariants
    are rooted in the SAME original K-dimensional map-patch perturbation box.
    """
    lowers = []
    uppers = []

    center = map_ctx["map_center"]
    radius = map_ctx["map_radius"]

    for s in range(0, C.shape[0], map_spec_batch):
        c = C[s:s + map_spec_batch].double()

        A_l, b_l, A_u, b_u = crown_affine_forms(
            prior_ops, c
        )

        for op in reversed(map_ctx["ops"]):
            A_l, db_l = op.backward(A_l, "lower")
            A_u, db_u = op.backward(A_u, "upper")
            b_l = b_l + db_l
            b_u = b_u + db_u

        lowers.append(
            center_radius_concretize(
                A_l, b_l, center, radius, "lower"
            )
        )
        uppers.append(
            center_radius_concretize(
                A_u, b_u, center, radius, "upper"
            )
        )

    return torch.cat(lowers), torch.cat(uppers)


def combined_state_bounds_direct(
    prior_ops,
    map_ctx,
    dim,
    map_spec_batch=8,
):
    eye = torch.eye(dim, dtype=torch.float64)
    return combined_crown_specs_direct(
        prior_ops,
        map_ctx,
        eye,
        map_spec_batch=map_spec_batch,
    )


def nominal_projection_variance_combined_lower(
    prior_ops,
    map_ctx,
    nominal_state,
    start,
    end,
    map_spec_batch=8,
):
    """Nominal-aligned variance lemma, bounded all the way to the map-patch roots."""
    z0 = nominal_state[start:end].double()
    n = int(z0.numel())

    if n <= 1:
        return 0.0, 0.0, 0.0

    centered = z0 - z0.mean()
    norm = torch.linalg.vector_norm(centered)

    if float(norm) <= 1e-30:
        return 0.0, 0.0, 0.0

    w = centered / norm

    C = torch.zeros(
        1,
        nominal_state.numel(),
        dtype=torch.float64,
    )
    C[0, start:end] = w

    l, u = combined_crown_specs_direct(
        prior_ops,
        map_ctx,
        C,
        map_spec_batch=map_spec_batch,
    )

    lo = float(l[0])
    hi = float(u[0])

    if lo > 0.0:
        minabs = lo
    elif hi < 0.0:
        minabs = -hi
    else:
        minabs = 0.0

    var_lb = (minabs * minabs) / float(n)
    var_lb = max(
        0.0,
        var_lb - (1e-14 + 1e-12 * abs(var_lb)),
    )
    return var_lb, lo, hi


def selected_pairwise_variance_combined_lower(
    prior_ops,
    map_ctx,
    nominal_state,
    start,
    end,
    pair_count=64,
    map_spec_batch=8,
):
    """Sound variance lower bound from selected high-energy pair differences.

    Var(z) = 1/n^2 * sum_{i<j} (z_i-z_j)^2.

    Any subset of the nonnegative terms gives a valid lower bound.  We choose
    the pairs with largest nominal |z_i-z_j|, then bound those differences
    directly to the original map-patch roots with custom CROWN.
    """
    z0 = nominal_state[start:end].double()
    n = int(z0.numel())

    if n <= 1 or pair_count <= 0:
        return 0.0, 0

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(abs(z0[i] - z0[j]))
            pairs.append((score, i, j))

    pairs.sort(reverse=True, key=lambda t: t[0])
    pairs = pairs[: min(pair_count, len(pairs))]

    if not pairs:
        return 0.0, 0

    rows = []
    state_dim = nominal_state.numel()
    for _, i, j in pairs:
        row = torch.zeros(state_dim, dtype=torch.float64)
        row[start + i] = 1.0
        row[start + j] = -1.0
        rows.append(row)

    C = torch.stack(rows)
    l, u = combined_crown_specs_direct(
        prior_ops,
        map_ctx,
        C,
        map_spec_batch=map_spec_batch,
    )

    minabs = torch.where(
        l > 0,
        l,
        torch.where(u < 0, -u, torch.zeros_like(l)),
    )

    var_lb = float(
        (minabs * minabs).sum() / float(n * n)
    )
    var_lb = max(
        0.0,
        var_lb - (1e-14 + 1e-12 * abs(var_lb)),
    )
    return var_lb, len(pairs)


def build_prior_crown(
    full_model,
    past_feat,
    feat_center,
    feat_l,
    feat_u,
    map_ctx,
    map_spec_batch,
    selected_pair_count,
    verbose=True,
):
    """Build prior operations and all required CROWN intermediate bounds."""
    if int(full_model.source_hidden.shape[1]) != 1:
        raise NotImplementedError(
            "This consolidated verifier currently targets the captured "
            "one-incoming-edge scene only."
        )

    root_nom = feat_center.reshape(-1).double()

    ops = []
    cur_nom = root_nom.clone()

    # Initial feature intervals are sound but are NOT used as an independent
    # CROWN root after the first prior operation.  All later bounds are
    # propagated directly to the four original map-pixel variables.
    cur_l = feat_l.reshape(-1).double()
    cur_u = feat_u.reshape(-1).double()

    diagnostics = []

    def recompute_bounds():
        nonlocal cur_l, cur_u
        dim = cur_nom.numel()

        cur_l, cur_u = combined_state_bounds_direct(
            ops,
            map_ctx,
            dim,
            map_spec_batch=map_spec_batch,
        )

        # very small numerical pad
        cur_l = cur_l - (1e-12 + 1e-10 * cur_l.abs())
        cur_u = cur_u + (1e-12 + 1e-10 * cur_u.abs())

        if bool((cur_l > cur_u).any()):
            raise RuntimeError("CROWN produced inverted prior intermediate bounds")

    def append_affine(W, b, name):
        nonlocal cur_nom
        op = AffineVecOp(W, b, name)
        ops.append(op)
        cur_nom = op.forward(cur_nom)
        recompute_bounds()
        diagnostics.append({
            "name": name,
            "type": "Affine",
            "dim": int(cur_nom.numel()),
            "max_width": float((cur_u - cur_l).max()),
        })

    def append_relu(start, end, name):
        nonlocal cur_nom
        op = ReLUVecOp(cur_l, cur_u, start, end, name)
        ops.append(op)
        cur_nom = op.forward(cur_nom)
        recompute_bounds()
        diagnostics.append({
            "name": name,
            "type": "ReLU",
            "dim": int(cur_nom.numel()),
            "max_width": float((cur_u - cur_l).max()),
            "unstable": int(
                (
                    (op.lower[start:end] < 0)
                    & (op.upper[start:end] > 0)
                ).sum()
            ),
        })

    def append_norm(layer, start, end, name):
        nonlocal cur_nom
        n = end - start

        direct_lb = float(
            exact_min_variance_box_1d(
                cur_l[start:end], cur_u[start:end]
            )
        )

        (
            projection_lb,
            projection_l,
            projection_u,
        ) = nominal_projection_variance_combined_lower(
            ops,
            map_ctx,
            cur_nom,
            start,
            end,
            map_spec_batch=map_spec_batch,
        )

        pair_lb, pair_used = (
            selected_pairwise_variance_combined_lower(
                ops,
                map_ctx,
                cur_nom,
                start,
                end,
                pair_count=selected_pair_count,
                map_spec_batch=map_spec_batch,
            )
        )

        var_lb = max(
            direct_lb,
            pair_lb,
            projection_lb,
        )
        var_ub = float(
            variance_upper_from_box(
                cur_l[start:end], cur_u[start:end]
            )
        )

        nominal_slice = cur_nom[start:end]
        nominal_var = float(
            ((nominal_slice - nominal_slice.mean()) ** 2).mean()
        )

        gamma, beta, eps_norm = _layernorm_params(layer)

        op = NormVecOp(
            cur_l,
            cur_u,
            gamma,
            beta,
            eps_norm,
            var_lb,
            var_ub,
            start,
            end,
            name,
            variance_meta={
                "nominal": nominal_var,
                "direct_lower": direct_lb,
                "pairwise_crown_lower": pair_lb,
                "projection_crown_lower": projection_lb,
                "projection_interval": [
                    projection_l,
                    projection_u,
                ],
                "used_lower": var_lb,
                "upper": var_ub,
            },
        )
        ops.append(op)
        cur_nom = op.forward(cur_nom)
        recompute_bounds()

        diag = {
            "name": name,
            "type": "CustomLayerNormCROWN",
            "dim": int(cur_nom.numel()),
            "normalized_dim": n,
            "variance_nominal": nominal_var,
            "variance_direct_lower": direct_lb,
            "variance_pairwise_crown_lower": pair_lb,
            "variance_pairwise_selected_count": pair_used,
            "variance_projection_crown_lower": projection_lb,
            "variance_projection_lower": projection_l,
            "variance_projection_upper": projection_u,
            "variance_lower_used": var_lb,
            "variance_upper": var_ub,
            "max_width": float((cur_u - cur_l).max()),
        }
        diagnostics.append(diag)

        if verbose:
            print(
                f"    {name}: var0={nominal_var:.8g}, "
                f"direct_lb={direct_lb:.8g}, "
                f"pair_crown_lb={pair_lb:.8g}({pair_used} pairs), "
                f"proj_crown_lb={projection_lb:.8g}, "
                f"proj=[{projection_l:.6g},{projection_u:.6g}], "
                f"used_lb={var_lb:.8g}, "
                f"post_width={diag['max_width']:.8g}"
            )

    def append_whole_layers(layers, prefix):
        for idx, layer in enumerate(layers):
            name = f"{prefix}[{idx}]"
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().double()
                b = (
                    layer.bias.detach().double()
                    if layer.bias is not None
                    else torch.zeros(W.shape[0], dtype=torch.float64)
                )
                append_affine(W, b, name)
            elif isinstance(layer, nn.LayerNorm):
                append_norm(layer, 0, cur_nom.numel(), name)
            elif isinstance(layer, nn.ReLU):
                append_relu(0, cur_nom.numel(), name)
            else:
                raise NotImplementedError(
                    f"{prefix}: unsupported layer {type(layer).__name__}"
                )

    # ------------------------------------------------------------------
    # prior_mlp_in. Input order:
    # [past64, map64, semantic2].
    # ------------------------------------------------------------------
    pin = list(full_model.prior_mlp_in)
    if not pin or not isinstance(pin[0], nn.Linear):
        raise RuntimeError("prior_mlp_in must start with Linear")

    past = past_feat.reshape(-1).double()
    sem_t = full_model.sem_target.reshape(-1).double()

    full_input = torch.cat([
        past,
        root_nom,
        sem_t,
    ])
    if full_input.numel() != pin[0].in_features:
        raise RuntimeError(
            f"prior_mlp_in input mismatch: {full_input.numel()} "
            f"vs {pin[0].in_features}"
        )

    map_start = past.numel()
    map_end = map_start + root_nom.numel()
    variable_idx = list(range(map_start, map_end))

    Wvar, beff = _fold_linear_with_constants(
        pin[0], variable_idx, full_input
    )
    append_affine(Wvar, beff, "prior_mlp_in[0]")
    append_whole_layers(pin[1:], "prior_mlp_in_tail")

    h_dim = cur_nom.numel()
    h_nom = cur_nom.clone()

    # ------------------------------------------------------------------
    # Edge MLP. Input:
    # [h_i, h_j_fixed, sem_i, sem_j, rel_trans].
    # Retain h in state while evaluating message: state becomes [h,msg].
    # ------------------------------------------------------------------
    edge = list(full_model.edge_mlp)
    if not edge or not isinstance(edge[0], nn.Linear):
        raise RuntimeError("edge_mlp must start with Linear")

    src = full_model.source_hidden.reshape(-1).double()
    sem_s = full_model.sem_sources.reshape(-1).double()
    rel = full_model.rel_trans.reshape(-1).double()

    edge_full = torch.cat([
        h_nom,
        src,
        sem_t,
        sem_s,
        rel,
    ])
    if edge_full.numel() != edge[0].in_features:
        raise RuntimeError(
            f"edge_mlp input mismatch: {edge_full.numel()} "
            f"vs {edge[0].in_features}"
        )

    hedge_idx = list(range(h_dim))
    Wedge_h, bedge = _fold_linear_with_constants(
        edge[0], hedge_idx, edge_full
    )
    msg_dim = Wedge_h.shape[0]

    Wstate = torch.zeros(
        h_dim + msg_dim,
        h_dim,
        dtype=torch.float64,
    )
    Wstate[:h_dim, :] = torch.eye(h_dim, dtype=torch.float64)
    Wstate[h_dim:, :] = Wedge_h

    bstate = torch.cat([
        torch.zeros(h_dim, dtype=torch.float64),
        bedge,
    ])
    append_affine(Wstate, bstate, "edge_state_init")

    edge_slice_start = h_dim
    edge_slice_end = h_dim + msg_dim

    for idx, layer in enumerate(edge[1:], start=1):
        name = f"edge_mlp[{idx}]"

        if isinstance(layer, nn.Linear):
            old_msg_dim = edge_slice_end - edge_slice_start
            new_msg_dim = layer.out_features

            W = torch.zeros(
                h_dim + new_msg_dim,
                h_dim + old_msg_dim,
                dtype=torch.float64,
            )
            W[:h_dim, :h_dim] = torch.eye(h_dim, dtype=torch.float64)
            W[h_dim:, h_dim:] = layer.weight.detach().double()

            b = torch.cat([
                torch.zeros(h_dim, dtype=torch.float64),
                (
                    layer.bias.detach().double()
                    if layer.bias is not None
                    else torch.zeros(new_msg_dim, dtype=torch.float64)
                ),
            ])
            append_affine(W, b, name)

            edge_slice_start = h_dim
            edge_slice_end = h_dim + new_msg_dim

        elif isinstance(layer, nn.LayerNorm):
            append_norm(
                layer,
                edge_slice_start,
                edge_slice_end,
                name,
            )
        elif isinstance(layer, nn.ReLU):
            append_relu(
                edge_slice_start,
                edge_slice_end,
                name,
            )
        else:
            raise NotImplementedError(
                f"edge_mlp: unsupported {type(layer).__name__}"
            )

    msg_dim = edge_slice_end - edge_slice_start

    # ------------------------------------------------------------------
    # Update MLP. Input [h,msg,sem_t].
    # With one incoming edge, coordinate-wise max is identity.
    # ------------------------------------------------------------------
    upd = list(full_model.update_mlp)
    if not upd or not isinstance(upd[0], nn.Linear):
        raise RuntimeError("update_mlp must start with Linear")

    state_nom = cur_nom
    upd_full = torch.cat([
        state_nom[:h_dim],
        state_nom[edge_slice_start:edge_slice_end],
        sem_t,
    ])
    if upd_full.numel() != upd[0].in_features:
        raise RuntimeError(
            f"update_mlp input mismatch: {upd_full.numel()} "
            f"vs {upd[0].in_features}"
        )

    variable_idx = list(range(h_dim + msg_dim))
    Wupd, bupd = _fold_linear_with_constants(
        upd[0], variable_idx, upd_full
    )
    append_affine(Wupd, bupd, "update_mlp[0]")
    append_whole_layers(upd[1:], "update_mlp_tail")

    # ------------------------------------------------------------------
    # Output MLP.
    # ------------------------------------------------------------------
    pout = list(full_model.prior_mlp_out)
    append_whole_layers(pout, "prior_mlp_out")

    if cur_nom.numel() != 64:
        raise RuntimeError(
            f"Expected 64 prior outputs, got {cur_nom.numel()}"
        )

    # v3.1 note: prior intermediate bounds are rooted directly in map_ctx's
    # original map-patch variables.  There is intentionally no root_l/root_u
    # feature-box root here.
    return {
        "ops": ops,
        "root_nom": root_nom,
        "output_nom": cur_nom,
        "output_l": cur_l,
        "output_u": cur_u,
        "diagnostics": diagnostics,
    }


# ============================================================================
# Map-CNN custom CROWN operations
# ============================================================================

class MapOp:
    def backward(self, A, side):
        raise NotImplementedError


class MapFeatureLinearOp(MapOp):
    def __init__(self, layer, input_shape):
        self.W = layer.weight.detach().double().clone()
        self.b = (
            layer.bias.detach().double().clone()
            if layer.bias is not None
            else torch.zeros(self.W.shape[0], dtype=torch.float64)
        )
        self.input_shape = tuple(input_shape)

    def backward(self, A, side):
        Aprev = A @ self.W
        bias = A @ self.b
        return Aprev.reshape((A.shape[0],) + self.input_shape), bias


class MapReLUOp(MapOp):
    def __init__(self, lower, upper, name):
        self.lower = lower.detach().double().clone()
        self.upper = upper.detach().double().clone()
        self.name = name

    def backward(self, A, side):
        l = self.lower
        u = self.upper

        pos = l >= 0
        neg = u <= 0
        unstable = ~(pos | neg)

        alpha_u = torch.zeros_like(l)
        beta_u = torch.zeros_like(l)
        alpha_l = torch.zeros_like(l)

        alpha_u[pos] = 1.0
        alpha_l[pos] = 1.0

        if bool(unstable.any()):
            lu = l[unstable]
            uu = u[unstable]
            au = uu / torch.clamp(uu - lu, min=1e-30)
            bu = -lu * au
            alpha_u[unstable] = au
            beta_u[unstable] = bu
            alpha_l[unstable] = (uu >= -lu).to(l.dtype)

        if side == "lower":
            choose_upper = A < 0
        else:
            choose_upper = A >= 0

        slope = torch.where(
            choose_upper,
            alpha_u.unsqueeze(0),
            alpha_l.unsqueeze(0),
        )
        intercept = torch.where(
            choose_upper,
            beta_u.unsqueeze(0),
            torch.zeros_like(beta_u).unsqueeze(0),
        )

        Aprev = A * slope
        bias = (A * intercept).reshape(A.shape[0], -1).sum(dim=1)
        return Aprev, bias


class MapNormOp(MapOp):
    def __init__(
        self,
        lower,
        upper,
        gamma,
        beta,
        eps_norm,
        var_lower,
        var_upper,
        name,
    ):
        self.lower = lower.detach().double().clone()
        self.upper = upper.detach().double().clone()
        self.gamma = gamma.detach().double().clone()
        self.beta = beta.detach().double().clone()
        self.eps_norm = float(eps_norm)
        self.var_lower = float(var_lower)
        self.var_upper = float(var_upper)
        self.name = name

    def backward(self, A, side):
        S = A.shape[0]
        Ain, bias = ratio_envelope(
            A.reshape(S, -1),
            self.lower.reshape(-1),
            self.upper.reshape(-1),
            self.gamma.reshape(-1),
            self.beta.reshape(-1),
            self.var_lower,
            self.var_upper,
            self.eps_norm,
            side,
        )
        return Ain.reshape_as(A), bias


class MapConvOp(MapOp):
    def __init__(self, layer, input_shape, output_shape, name):
        self.weight = layer.weight.detach().double().clone()
        self.bias = (
            layer.bias.detach().double().clone()
            if layer.bias is not None
            else None
        )
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.name = name

    def backward(self, A, side):
        if self.bias is None:
            bias = torch.zeros(A.shape[0], dtype=torch.float64)
        else:
            bias = (
                A * self.bias.view(1, -1, 1, 1)
            ).reshape(A.shape[0], -1).sum(dim=1)

        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        in_h, in_w = self.input_shape[-2:]
        out_h, out_w = self.output_shape[-2:]
        kh, kw = self.weight.shape[-2:]

        op_h = (
            in_h
            - (out_h - 1) * stride[0]
            + 2 * padding[0]
            - dilation[0] * (kh - 1)
            - 1
        )
        op_w = (
            in_w
            - (out_w - 1) * stride[1]
            + 2 * padding[1]
            - dilation[1] * (kw - 1)
            - 1
        )

        if not (0 <= op_h < stride[0] and 0 <= op_w < stride[1]):
            raise RuntimeError(
                f"{self.name}: invalid output_padding {(op_h,op_w)}"
            )

        Aprev = F.conv_transpose2d(
            A,
            self.weight,
            None,
            stride=stride,
            padding=padding,
            output_padding=(op_h, op_w),
            groups=self.groups,
            dilation=dilation,
        )
        return Aprev, bias


def groupnorm_variance_bounds(
    nominal,
    lower,
    upper,
    deriv_l,
    deriv_u,
    halfwidth,
):
    dims = (1, 2, 3)
    ddims = (2, 3, 4)

    mean0 = nominal.mean(dim=dims, keepdim=True)
    centered0 = nominal - mean0
    var0 = (centered0 * centered0).mean(dim=dims, keepdim=True)

    direct_lb = exact_min_variance_box_1d(lower, upper)
    direct_ub = variance_upper_from_box(lower, upper)

    mean_l = lower.mean(dim=dims, keepdim=True)
    mean_u = upper.mean(dim=dims, keepdim=True)
    centered_l = lower - mean_u
    centered_u = upper - mean_l

    prod_candidates = []
    # d Var/dq = 2/N sum centered * dx/dq
    a_l = centered_l.unsqueeze(0)
    a_u = centered_u.unsqueeze(0)
    b_l = deriv_l
    b_u = deriv_u

    p1 = a_l * b_l
    p2 = a_l * b_u
    p3 = a_u * b_l
    p4 = a_u * b_u
    prod_l = torch.minimum(torch.minimum(p1, p2), torch.minimum(p3, p4))
    prod_u = torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))

    dvar_l = 2.0 * prod_l.mean(dim=ddims, keepdim=True)
    dvar_u = 2.0 * prod_u.mean(dim=ddims, keepdim=True)

    k = deriv_l.shape[0]
    h = halfwidth.reshape(k, 1, 1, 1, 1)
    rad = (
        h * torch.maximum(dvar_l.abs(), dvar_u.abs())
    ).sum(dim=0)

    mvt_lb = torch.clamp(var0 - rad, min=0.0).reshape(-1)[0]
    mvt_ub = (var0 + rad).reshape(-1)[0]

    used_lb = max(float(direct_lb), float(mvt_lb))
    used_ub = min(float(direct_ub), float(mvt_ub))
    used_ub = max(used_ub, used_lb)

    return {
        "nominal": float(var0.reshape(-1)[0]),
        "direct_lower": float(direct_lb),
        "mvt_lower": float(mvt_lb),
        "used_lower": used_lb,
        "direct_upper": float(direct_ub),
        "mvt_upper": float(mvt_ub),
        "used_upper": used_ub,
    }


def map_nominal_projection_variance_lower(
    ops,
    map_center,
    map_radius,
    nominal_state,
):
    """Sound GroupNorm variance lower bound rooted in the map-patch variables.

    For GroupNorm(1,C), all C*H*W values share one normalization group.
    Let c0 = z0 - mean(z0) and w = c0 / ||c0||. Because w is orthogonal
    to the all-ones vector,

        Var(z) = ||z - mean(z)||^2 / n >= (w^T z)^2 / n.

    We bound w^T z directly through all map operations accumulated so far,
    down to the original patch raster variables. This preserves correlations
    that are lost by plain elementwise interval propagation.
    """
    z0 = nominal_state.reshape(-1).double()
    n = int(z0.numel())

    if n <= 1:
        return 0.0, 0.0, 0.0

    centered = z0 - z0.mean()
    norm = torch.linalg.vector_norm(centered)

    if float(norm) <= 1e-30:
        return 0.0, 0.0, 0.0

    w = centered / norm
    A = w.reshape((1,) + tuple(nominal_state.shape[1:]))

    A_l, b_l = map_backward_affine(ops, A, "lower")
    A_u, b_u = map_backward_affine(ops, A, "upper")

    l = center_radius_concretize(
        A_l, b_l, map_center, map_radius, "lower"
    )
    u = center_radius_concretize(
        A_u, b_u, map_center, map_radius, "upper"
    )

    lo = float(l[0])
    hi = float(u[0])

    if lo > 0.0:
        minabs = lo
    elif hi < 0.0:
        minabs = -hi
    else:
        minabs = 0.0

    var_lb = (minabs * minabs) / float(n)
    var_lb = max(
        0.0,
        var_lb - (1e-14 + 1e-12 * abs(var_lb)),
    )
    return var_lb, lo, hi


def map_state_crown_bounds(
    ops,
    state_shape,
    map_center,
    map_radius,
    spec_batch=8,
):
    """Coordinatewise CROWN bounds for a map intermediate state.

    The specs are identities at the requested state and are propagated
    backward through all accumulated map operations to the original map-patch
    roots.  This is deliberately used only for late/small map states
    (especially Conv6/GN6 and the final 64-D map feature), where the number
    of coordinate specs is manageable.
    """
    dim = 1
    for d in state_shape:
        dim *= int(d)

    lower_chunks = []
    upper_chunks = []

    for s in range(0, dim, spec_batch):
        e = min(dim, s + spec_batch)
        rows = e - s

        C = torch.zeros(
            (rows, dim),
            dtype=torch.float64,
        )
        local = torch.arange(rows)
        global_idx = torch.arange(s, e)
        C[local, global_idx] = 1.0
        C = C.reshape((rows,) + tuple(state_shape))

        A_l, b_l = map_backward_affine(ops, C, "lower")
        A_u, b_u = map_backward_affine(ops, C, "upper")

        l = center_radius_concretize(
            A_l,
            b_l,
            map_center,
            map_radius,
            "lower",
        )
        u = center_radius_concretize(
            A_u,
            b_u,
            map_center,
            map_radius,
            "upper",
        )

        lower_chunks.append(l.detach())
        upper_chunks.append(u.detach())

    lower = torch.cat(lower_chunks).reshape((1,) + tuple(state_shape))
    upper = torch.cat(upper_chunks).reshape((1,) + tuple(state_shape))

    # Small outward pad for floating-point conservatism.
    scale = torch.maximum(
        torch.ones_like(lower),
        torch.maximum(lower.abs(), upper.abs()),
    )
    pad = 1e-12 + 1e-10 * scale
    return lower - pad, upper + pad



def build_map_crown_context(
    map_encoder,
    base_map,
    row0,
    row1,
    col0,
    col1,
    root_center,
    root_halfwidth,
    abs_slack,
    rel_slack,
    verbose=True,
):
    """Forward sound map bounds and record custom-CROWN operations.

    Root order is channel-major over the selected patch.
    """
    dtype = torch.float64

    enc = map_encoder.double().eval()
    base = base_map.double().clone()

    ph = int(row1 - row0 + 1)
    pw = int(col1 - col0 + 1)
    root_dim = int(root_center.numel())

    expected_root_dim = 4 * ph * pw
    if root_dim != expected_root_dim:
        raise RuntimeError(
            f"Patch root dimension mismatch: got {root_dim}, "
            f"expected {expected_root_dim} = 4*{ph}*{pw}"
        )

    patch_shape = (1, 4, ph, pw)

    nominal = base.clone()
    nominal[:, :, row0:row1 + 1, col0:col1 + 1] = (
        root_center.reshape(*patch_shape)
    )

    lower = nominal.clone()
    upper = nominal.clone()
    lower[:, :, row0:row1 + 1, col0:col1 + 1] -= (
        root_halfwidth.reshape(*patch_shape)
    )
    upper[:, :, row0:row1 + 1, col0:col1 + 1] += (
        root_halfwidth.reshape(*patch_shape)
    )

    deriv_l = torch.zeros(
        (root_dim,) + tuple(nominal.shape),
        dtype=dtype,
    )
    deriv_u = torch.zeros_like(deriv_l)

    per_channel = ph * pw
    for q in range(root_dim):
        ch = q // per_channel
        rem = q % per_channel
        rr = rem // pw
        cc = rem % pw
        deriv_l[q, 0, ch, row0 + rr, col0 + cc] = 1.0
        deriv_u[q, 0, ch, row0 + rr, col0 + cc] = 1.0

    map_center = base.clone()
    map_center[:, :, row0:row1 + 1, col0:col1 + 1] = (
        root_center.reshape(*patch_shape)
    )
    map_radius = torch.zeros_like(map_center)
    map_radius[:, :, row0:row1 + 1, col0:col1 + 1] = (
        root_halfwidth.reshape(*patch_shape)
    )

    ops = []
    diagnostics = []

    for block in range(6):
        conv = enc.net[3 * block]
        gn = enc.net[3 * block + 1]
        relu = enc.net[3 * block + 2]

        input_shape = tuple(nominal.shape[1:])

        nominal_conv = conv(nominal)
        conv_l, conv_u = conv_value_interval(conv, lower, upper)
        dconv_l, dconv_u = conv_deriv_interval(conv, deriv_l, deriv_u)

        conv_l, conv_u = outward(
            conv_l, conv_u, abs_slack, rel_slack
        )
        dconv_l, dconv_u = outward(
            dconv_l, dconv_u, abs_slack, rel_slack
        )

        conv_l, conv_u, _ = mvt_value_box(
            nominal_conv,
            conv_l,
            conv_u,
            dconv_l,
            dconv_u,
            root_halfwidth,
            abs_slack,
            rel_slack,
        )

        output_shape = tuple(nominal_conv.shape[1:])
        ops.append(
            MapConvOp(
                conv,
                input_shape,
                output_shape,
                f"conv{block+1}",
            )
        )

        # For the final small Conv6 state (128x2x2 = 512 coordinates),
        # tighten the coordinate box directly by backward CROWN to the
        # original map-patch variables. Earlier blocks remain on the fast
        # sensitivity propagation because their states are very large.
        late_crown_tightened = (block == 5)

        if late_crown_tightened:
            conv_l_crown, conv_u_crown = map_state_crown_bounds(
                ops,
                tuple(nominal_conv.shape[1:]),
                map_center[0],
                map_radius[0],
                spec_batch=8,
            )
            conv_l = torch.maximum(conv_l, conv_l_crown)
            conv_u = torch.minimum(conv_u, conv_u_crown)

            # In case tiny floating point mismatches invert an endpoint,
            # fall back to the sound union endpoint.
            bad = conv_l > conv_u
            if bool(bad.any()):
                midpoint = 0.5 * (conv_l + conv_u)
                conv_l = torch.where(bad, midpoint - 1e-10, conv_l)
                conv_u = torch.where(bad, midpoint + 1e-10, conv_u)

        var_meta = groupnorm_variance_bounds(
            nominal_conv,
            conv_l,
            conv_u,
            dconv_l,
            dconv_u,
            root_halfwidth,
        )

        (
            map_projection_lb,
            map_projection_l,
            map_projection_u,
        ) = map_nominal_projection_variance_lower(
            ops,
            map_center[0],
            map_radius[0],
            nominal_conv,
        )

        var_meta["projection_crown_lower"] = map_projection_lb
        var_meta["projection_interval"] = [
            map_projection_l,
            map_projection_u,
        ]
        var_meta["used_lower"] = max(
            var_meta["used_lower"],
            map_projection_lb,
        )

        if late_crown_tightened:
            tight_box_lb = float(exact_min_variance_box_1d(
                conv_l.detach(),
                conv_u.detach(),
            ))
            tight_box_ub = float(variance_upper_from_box(
                conv_l.detach(),
                conv_u.detach(),
            ))
            var_meta["late_crown_box_lower"] = tight_box_lb
            var_meta["late_crown_box_upper"] = tight_box_ub
            var_meta["used_lower"] = max(
                var_meta["used_lower"],
                tight_box_lb,
            )
            var_meta["used_upper"] = min(
                var_meta["used_upper"],
                tight_box_ub,
            )

        var_meta["used_upper"] = max(
            var_meta["used_upper"],
            var_meta["used_lower"],
        )

        (
            nominal_gn,
            gn_l_fast,
            gn_u_fast,
            dgn_l,
            dgn_u,
            _,
        ) = groupnorm1_sensitivity(
            gn,
            nominal_conv,
            conv_l,
            conv_u,
            dconv_l,
            dconv_u,
            root_halfwidth,
            abs_slack,
            rel_slack,
        )

        C, H, W = nominal_conv.shape[1:]
        gamma = (
            gn.weight.detach().double()
            .view(C, 1, 1)
            .expand(C, H, W)
        )
        beta = (
            gn.bias.detach().double()
            .view(C, 1, 1)
            .expand(C, H, W)
        )

        ops.append(
            MapNormOp(
                conv_l[0],
                conv_u[0],
                gamma,
                beta,
                gn.norm.eps,
                var_meta["used_lower"],
                var_meta["used_upper"],
                f"gn{block+1}",
            )
        )

        if late_crown_tightened:
            # Now that the custom GN6 operator is in the CROWN graph,
            # directly bound all 512 GN6 outputs to the original 36 roots.
            gn_l_crown, gn_u_crown = map_state_crown_bounds(
                ops,
                tuple(nominal_gn.shape[1:]),
                map_center[0],
                map_radius[0],
                spec_batch=8,
            )
            gn_l = torch.maximum(gn_l_fast, gn_l_crown)
            gn_u = torch.minimum(gn_u_fast, gn_u_crown)

            bad = gn_l > gn_u
            if bool(bad.any()):
                midpoint = 0.5 * (gn_l + gn_u)
                gn_l = torch.where(bad, midpoint - 1e-10, gn_l)
                gn_u = torch.where(bad, midpoint + 1e-10, gn_u)

            nominal_relu = relu(nominal_gn)
            relu_l = torch.relu(gn_l)
            relu_u = torch.relu(gn_u)

            # Derivative bounds are no longer needed after block 6.
            drelu_l = dgn_l
            drelu_u = dgn_u
        else:
            gn_l = gn_l_fast
            gn_u = gn_u_fast
            nominal_relu = relu(nominal_gn)
            (
                relu_l,
                relu_u,
                drelu_l,
                drelu_u,
                _,
            ) = relu_propagate(
                nominal_gn,
                nominal_relu,
                gn_l,
                gn_u,
                dgn_l,
                dgn_u,
                root_halfwidth,
                abs_slack,
                rel_slack,
            )

        ops.append(
            MapReLUOp(
                gn_l[0],
                gn_u[0],
                f"relu{block+1}",
            )
        )

        diagnostics.append({
            "block": block + 1,
            "variance": var_meta,
            "late_crown_tightened": late_crown_tightened,
            "post_relu_max_width": float((relu_u - relu_l).max()),
        })

        if verbose:
            extra = ""
            if late_crown_tightened:
                extra = (
                    f", crown_box_var=["
                    f"{var_meta.get('late_crown_box_lower', 0):.8g},"
                    f"{var_meta.get('late_crown_box_upper', 0):.8g}]"
                )
            print(
                f"    map GN{block+1}: "
                f"var0={var_meta['nominal']:.8g}, "
                f"direct/mvt_lb={max(var_meta['direct_lower'], var_meta['mvt_lower']):.8g}, "
                f"proj_crown_lb={var_meta['projection_crown_lower']:.8g}, "
                f"proj=[{var_meta['projection_interval'][0]:.6g},"
                f"{var_meta['projection_interval'][1]:.6g}], "
                f"used_lb={var_meta['used_lower']:.8g}, "
                f"ub={var_meta['used_upper']:.8g}, "
                f"post_relu_width={diagnostics[-1]['post_relu_max_width']:.8g}"
                f"{extra}"
            )

        nominal = nominal_relu
        lower, upper = relu_l, relu_u
        deriv_l, deriv_u = drelu_l, drelu_u

    nominal_flat = nominal.reshape(1, -1)
    lower_flat = lower.reshape(1, -1)
    upper_flat = upper.reshape(1, -1)
    dflat_l = deriv_l.reshape(root_dim, 1, -1)
    dflat_u = deriv_u.reshape(root_dim, 1, -1)

    feature = enc.feature
    feat_nom = feature(nominal_flat)
    feat_l, feat_u = map_linear_value_interval(
        feature, lower_flat, upper_flat
    )
    dfeat_l, dfeat_u = map_linear_deriv_interval(
        feature, dflat_l, dflat_u
    )

    feat_l, feat_u = outward(
        feat_l, feat_u, abs_slack, rel_slack
    )
    dfeat_l, dfeat_u = outward(
        dfeat_l, dfeat_u, abs_slack, rel_slack
    )

    feat_l, feat_u, _ = mvt_value_box(
        feat_nom,
        feat_l,
        feat_u,
        dfeat_l,
        dfeat_u,
        root_halfwidth,
        abs_slack,
        rel_slack,
    )

    ops.append(
        MapFeatureLinearOp(
            feature,
            tuple(nominal.shape[1:]),
        )
    )

    # Final map feature has only 64 coordinates. Bound it directly through
    # the complete Conv/GN/ReLU/Linear map CROWN graph to the original patch.
    feat_l_crown, feat_u_crown = map_state_crown_bounds(
        ops,
        (64,),
        map_center[0],
        map_radius[0],
        spec_batch=8,
    )
    feat_l_direct = torch.maximum(
        feat_l.reshape(1, 64),
        feat_l_crown.reshape(1, 64),
    )
    feat_u_direct = torch.minimum(
        feat_u.reshape(1, 64),
        feat_u_crown.reshape(1, 64),
    )

    bad = feat_l_direct > feat_u_direct
    if bool(bad.any()):
        midpoint = 0.5 * (feat_l_direct + feat_u_direct)
        feat_l_direct = torch.where(
            bad, midpoint - 1e-10, feat_l_direct
        )
        feat_u_direct = torch.where(
            bad, midpoint + 1e-10, feat_u_direct
        )

    if verbose:
        print(
            f"    final map feature direct-CROWN width="
            f"{float((feat_u_direct-feat_l_direct).max()):.8g}"
        )

    return {
        "ops": ops,
        "feature_nominal": feat_nom.reshape(-1),
        "feature_lower": feat_l_direct.reshape(-1),
        "feature_upper": feat_u_direct.reshape(-1),
        "map_center": map_center[0],
        "map_radius": map_radius[0],
        "diagnostics": diagnostics,
        "root_dim": root_dim,
        "patch_shape": [4, ph, pw],
        "patch_bounds": [row0, row1, col0, col1],
    }

def map_backward_affine(ops, A, side):
    A = A.double()
    bias = torch.zeros(A.shape[0], dtype=torch.float64)

    for op in reversed(ops):
        A, db = op.backward(A, side)
        bias = bias + db

    return A, bias


# ============================================================================
# One complete leaf
# ============================================================================

def verify_leaf(
    leaf_id,
    depth,
    root_center,
    root_halfwidth,
    full_model,
    past_feat,
    base_map,
    row0,
    row1,
    col0,
    col1,
    global_nominal,
    args,
    verbose=False,
):
    t0 = time.time()

    try:
        map_ctx = build_map_crown_context(
            full_model.map_encoder,
            base_map,
            row0,
            row1,
            col0,
            col1,
            root_center,
            root_halfwidth,
            args.abs_slack,
            args.rel_slack,
            verbose=verbose,
        )

        prior_ctx = build_prior_crown(
            full_model,
            past_feat,
            map_ctx["feature_nominal"],
            map_ctx["feature_lower"],
            map_ctx["feature_upper"],
            map_ctx,
            args.map_spec_batch,
            args.selected_pair_count,
            verbose=verbose,
        )

        prior_nom_err = float(
            (
                prior_ctx["output_nom"].float()
                - global_nominal.reshape(-1)
            ).abs().max()
        )

        C = torch.eye(64, dtype=torch.float64)
        lb64, ub64 = combined_crown_specs_direct(
            prior_ctx["ops"],
            map_ctx,
            C,
            map_spec_batch=args.map_spec_batch,
        )

        lb = lb64.float().reshape(1, 64)
        ub = ub64.float().reshape(1, 64)

        nominal = global_nominal.float()
        ok = (
            (lb >= nominal - args.delta)
            & (ub <= nominal + args.delta)
        )
        dev = torch.maximum(
            nominal - lb,
            ub - nominal,
        )
        worst_dev, worst_idx = torch.max(dev, dim=1)

        min_prior_var = min(
            (
                d["variance_lower_used"]
                for d in prior_ctx["diagnostics"]
                if d["type"] == "CustomLayerNormCROWN"
            ),
            default=float("inf"),
        )

        return {
            "status": "BOUNDED",
            "leaf_id": leaf_id,
            "depth": depth,
            "center": root_center.tolist(),
            "halfwidth": root_halfwidth.tolist(),
            "lb": lb,
            "ub": ub,
            "all_certified": bool(ok.all()),
            "certified_count": int(ok.sum()),
            "worst_deviation": float(worst_dev[0]),
            "worst_index": int(worst_idx[0]),
            "prior_nominal_equiv_error": prior_nom_err,
            "min_prior_variance_lower": float(min_prior_var),
            "feature_max_width": float(
                (
                    map_ctx["feature_upper"]
                    - map_ctx["feature_lower"]
                ).max()
            ),
            "map_diagnostics": map_ctx["diagnostics"],
            "prior_diagnostics": prior_ctx["diagnostics"],
            "elapsed_sec": time.time() - t0,
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "leaf_id": leaf_id,
            "depth": depth,
            "center": root_center.tolist(),
            "halfwidth": root_halfwidth.tolist(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - t0,
        }


# ============================================================================
# Main / optional K-D patch-input BaB
# ============================================================================

def split_box(center, halfwidth):
    q = int(torch.argmax(halfwidth))
    h = halfwidth.clone()
    h[q] *= 0.5

    left = center.clone()
    right = center.clone()
    left[q] -= h[q]
    right[q] += h[q]

    return q, (left, h.clone()), (right, h.clone())


def output_name(i):
    return f"mu_{i}" if i < 32 else f"logvar_{i-32}"


def main(args):
    torch.set_num_threads(args.threads)

    cap = torch.load(
        args.capture,
        map_location="cpu",
        weights_only=True,
    )
    full_model, incoming, _ = build_model(args.ckpt, cap)
    full_model = full_model.double().eval()

    if int(incoming.numel()) != 1:
        raise NotImplementedError(
            f"Current consolidated verifier expects one incoming edge; "
            f"capture has {incoming.numel()}."
        )

    past0 = cap["target_past_input"].reshape(1, -1).double()
    map0 = cap["target_map_obs"].unsqueeze(0).double()

    target = int(cap["target_agent"])
    recorded = cap["original_output"][target].reshape(1, 64).double()

    _, channels, height, width = map0.shape
    if channels != 4:
        raise RuntimeError(f"Expected 4 map channels, got {channels}")

    row = (height - 1) // 2 if args.row is None else args.row
    col = (width - 1) // 2 if args.col is None else args.col

    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be a positive odd integer")

    half = args.patch_size // 2
    row0, row1 = row - half, row + half
    col0, col1 = col - half, col + half

    if row0 < 0 or col0 < 0 or row1 >= height or col1 >= width:
        raise ValueError(
            f"Requested patch centered at ({row},{col}) with "
            f"patch_size={args.patch_size} exceeds map bounds "
            f"{height}x{width}"
        )

    root_dim = 4 * args.patch_size * args.patch_size

    with torch.no_grad():
        past_feat, map_feat = full_model.encode_branches(past0, map0)
        full_nominal = full_model(past0, map0)

    map_err = float(
        (
            map_feat.float()
            - cap["target_map_feat"].reshape(1, 64).float()
        ).abs().max()
    )
    full_err = float(
        (
            full_nominal.float()
            - recorded.float()
        ).abs().max()
    )

    print("=" * 80)
    print("STRIVE STAGE D2-GENERALIZED PATCH LATE-MAP CROWN TIGHTENING")
    print("=" * 80)
    print(f"target agent              : {target}")
    print(f"incoming edges            : {incoming.numel()}")
    print(f"patch center              : ({row},{col})")
    print(f"patch rows                : {row0}..{row1}")
    print(f"patch cols                : {col0}..{col1}")
    print(f"patch size                : {args.patch_size}x{args.patch_size}")
    print(f"perturbed values          : {root_dim}")
    print(f"epsilon                   : {args.epsilon}")
    print(f"delta                     : {args.delta}")
    print(f"map equivalence error     : {map_err:.8g}")
    print(f"full equivalence error    : {full_err:.8g}")
    print(f"max BaB depth             : {args.max_depth}")
    print(f"max evaluated nodes       : {args.max_nodes}")
    print(f"prior CROWN spec batch    : {args.prior_spec_batch}")
    print(f"map CROWN spec batch      : {args.map_spec_batch}")
    print(f"pairwise variance batch   : {args.pair_batch}")
    print(f"selected pair count       : {args.selected_pair_count}")
    print(f"prior intermediate root   : original {root_dim} map-patch scalars")

    if max(map_err, full_err) > args.equiv_tol:
        raise RuntimeError(
            "Reconstructed verifier model does not match capture within "
            f"{args.equiv_tol}; no verification claim."
        )

    root_center = (
        map0[0, :, row0:row1 + 1, col0:col1 + 1]
        .contiguous()
        .reshape(-1)
        .clone()
    )
    root_halfwidth = torch.full(
        (root_dim,),
        float(args.epsilon),
        dtype=torch.float64,
    )

    print("\nROOT CUSTOM-NORMALIZATION CROWN")
    print("-" * 80)

    root = verify_leaf(
        0,
        0,
        root_center,
        root_halfwidth,
        full_model,
        past_feat,
        map0,
        row0,
        row1,
        col0,
        col1,
        full_nominal.float(),
        args,
        verbose=True,
    )

    if root["status"] == "BOUNDED":
        print(
            f"  prior nominal equiv error : "
            f"{root['prior_nominal_equiv_error']:.8g}"
        )
        print(
            f"  map feature max width      : "
            f"{root['feature_max_width']:.8g}"
        )
        print(
            f"  min prior variance lower   : "
            f"{root['min_prior_variance_lower']:.8g}"
        )
        print(
            f"  certified outputs          : "
            f"{root['certified_count']}/64"
        )
        print(
            f"  worst output               : "
            f"{output_name(root['worst_index'])}"
        )
        print(
            f"  worst max deviation        : "
            f"{root['worst_deviation']:.8g}"
        )
    else:
        print(
            f"  ERROR {root['exception_type']}: "
            f"{root['exception']}"
        )
        print(root["traceback"])

    terminal = []
    queue = []
    evaluated = 1
    next_id = 1

    def priority(node):
        if node["status"] == "ERROR":
            return -1e30
        return -float(node["worst_deviation"])

    if (
        root["status"] == "BOUNDED"
        and root["all_certified"]
    ):
        terminal.append(root)
    elif args.max_depth == 0 or args.max_nodes == 1:
        terminal.append(root)
    else:
        heapq.heappush(queue, (priority(root), 0, root))

    print(f"\n{root_dim}-D INPUT BRANCH-AND-BOUND")
    print("-" * 80)

    while queue and evaluated < args.max_nodes:
        _, _, node = heapq.heappop(queue)

        if node["depth"] >= args.max_depth:
            terminal.append(node)
            continue

        center = torch.tensor(node["center"], dtype=torch.float64)
        halfwidth = torch.tensor(node["halfwidth"], dtype=torch.float64)

        split_dim, left, right = split_box(center, halfwidth)

        if evaluated + 2 > args.max_nodes:
            terminal.append(node)
            break

        for child_center, child_halfwidth in (left, right):
            child_id = next_id
            next_id += 1

            child = verify_leaf(
                child_id,
                node["depth"] + 1,
                child_center,
                child_halfwidth,
                full_model,
                past_feat,
                map0,
                row,
                col,
                full_nominal.float(),
                args,
                verbose=False,
            )
            evaluated += 1

            if child["status"] == "BOUNDED":
                print(
                    f"  leaf={child_id:3d} "
                    f"depth={child['depth']} "
                    f"split=x{split_dim} "
                    f"minVar={child['min_prior_variance_lower']:.6g} "
                    f"certified={child['certified_count']:2d}/64 "
                    f"worst_dev={child['worst_deviation']:.6g}"
                )
            else:
                print(
                    f"  leaf={child_id:3d} "
                    f"depth={child['depth']} "
                    f"ERROR {child['exception_type']}: "
                    f"{child['exception']}"
                )

            if (
                child["status"] == "BOUNDED"
                and child["all_certified"]
            ):
                terminal.append(child)
            elif child["depth"] >= args.max_depth:
                terminal.append(child)
            else:
                heapq.heappush(
                    queue,
                    (priority(child), child["leaf_id"], child),
                )

    while queue:
        _, _, node = heapq.heappop(queue)
        terminal.append(node)

    all_bounded = all(
        n["status"] == "BOUNDED"
        for n in terminal
    )
    any_errors = any(
        n["status"] == "ERROR"
        for n in terminal
    )
    all_certified = (
        len(terminal) > 0
        and all(
            n["status"] == "BOUNDED"
            and n.get("all_certified", False)
            for n in terminal
        )
    )

    output_rows = []

    if all_bounded:
        lbs = torch.cat([n["lb"] for n in terminal], dim=0)
        ubs = torch.cat([n["ub"] for n in terminal], dim=0)

        global_lb = lbs.min(dim=0, keepdim=True).values
        global_ub = ubs.max(dim=0, keepdim=True).values

        for i in range(64):
            nominal = float(full_nominal[0, i])
            lo = float(global_lb[0, i])
            hi = float(global_ub[0, i])
            max_dev = max(nominal - lo, hi - nominal)

            ok = (
                lo >= nominal - args.delta
                and hi <= nominal + args.delta
            )

            output_rows.append({
                "output": output_name(i),
                "index": i,
                "nominal": nominal,
                "certified_lower": lo,
                "certified_upper": hi,
                "max_deviation": max_dev,
                "delta": args.delta,
                "verdict": "CERTIFIED" if ok else "NOT_PROVED",
            })

    if all_certified:
        status = "CERTIFIED"
    elif any_errors:
        status = "NO_VERIFICATION_RESULT"
    elif all_bounded:
        status = "NOT_PROVED"
    else:
        status = "NO_VERIFICATION_RESULT"

    print("\n" + "=" * 80)
    print("FINAL STAGE D2-GENERALIZED PATCH LATE-MAP CROWN TIGHTENING RESULT")
    print("=" * 80)
    print(f"status                     : {status}")
    print(f"evaluated BaB nodes        : {evaluated}")
    print(f"terminal regions           : {len(terminal)}")

    if output_rows:
        certified = sum(
            r["verdict"] == "CERTIFIED"
            for r in output_rows
        )
        worst = max(
            output_rows,
            key=lambda r: r["max_deviation"],
        )

        print(
            f"globally certified outputs : "
            f"{certified}/64"
        )
        print(
            f"worst output               : "
            f"{worst['output']}"
        )
        print(
            f"worst global deviation     : "
            f"{worst['max_deviation']:.8g}"
        )
        print(
            f"allowed delta              : "
            f"{args.delta:.8g}"
        )

    if any_errors:
        print("\nUnresolved error leaves:")
        for n in terminal:
            if n["status"] == "ERROR":
                print(
                    f"  leaf={n['leaf_id']} "
                    f"depth={n['depth']} "
                    f"{n['exception_type']}: "
                    f"{n['exception']}"
                )

    serial_terminal = []
    for n in terminal:
        item = {
            k: v
            for k, v in n.items()
            if k not in {"lb", "ub", "traceback"}
        }
        serial_terminal.append(item)

    result = {
        "stage": "D2-generalized-map-patch-late-map-crown-tightening",
        "status": status,
        "scope": {
            "target_agent": target,
            "incoming_edges": int(incoming.numel()),
            "patch_center": [row, col],
            "patch_bounds": [row0, row1, col0, col1],
            "patch_size": args.patch_size,
            "perturbed_values": root_dim,
            "epsilon": args.epsilon,
            "delta": args.delta,
            "held_fixed": [
                "past input",
                "visibility and vehicle attributes",
                "semantic class",
                "graph topology",
                "relative poses",
                "all non-target agents",
                "all map pixels outside the selected patch",
            ],
        },
        "equivalence": {
            "map_feature_max_abs_error": map_err,
            "full_target_max_abs_error": full_err,
            "tolerance": args.equiv_tol,
        },
        "method": {
            "name": "model-specific custom-normalization backward CROWN",
            "normalization_relaxation": (
                "arbitrary-A ratio envelope for a^T LayerNorm/GroupNorm, "
                "preserving affine dependence backward"
            ),
            "layernorm_variance": (
                "max(exact box min-variance, CROWN pairwise-difference "
                "variance invariant)"
            ),
            "groupnorm_variance": (
                "max(exact box min-variance, K-root patch MVT variance invariant)"
            ),
            "relu": "standard adaptive CROWN lower relaxation",
            "branch_and_bound": (
                "optional coordinate splitting over only the selected map-patch values"
            ),
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
        },
        "evaluated_nodes": evaluated,
        "terminal_regions": serial_terminal,
        "outputs": output_rows,
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2))

    if output_rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(output_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(output_rows)

    print(f"\nwrote {args.out_json}")
    if output_rows:
        print(f"wrote {args.out_csv}")

    if status == "CERTIFIED":
        return 0
    if status == "NOT_PROVED":
        return 2
    return 3


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", required=True)
    p.add_argument("--capture", required=True)

    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--delta", type=float, default=0.05)

    p.add_argument("--row", type=int, default=None)
    p.add_argument("--col", type=int, default=None)
    p.add_argument(
        "--patch-size",
        type=int,
        default=3,
        help="Odd square patch side length; default 3 gives 36 uncertain values.",
    )

    # Start modestly.  If root is not certified, the same command can split.
    p.add_argument("--max-depth", type=int, default=0)
    p.add_argument("--max-nodes", type=int, default=1)

    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--equiv-tol", type=float, default=1e-4)

    p.add_argument("--prior-spec-batch", type=int, default=128)
    p.add_argument("--map-spec-batch", type=int, default=8)
    p.add_argument("--pair-batch", type=int, default=512)
    p.add_argument(
        "--selected-pair-count",
        type=int,
        default=64,
        help=(
            "Number of largest nominal LayerNorm coordinate differences "
            "used in the direct-to-patch-root pairwise variance invariant."
        ),
    )

    p.add_argument("--abs-slack", type=float, default=1e-12)
    p.add_argument("--rel-slack", type=float, default=1e-10)

    p.add_argument(
        "--out-json",
        default="prior_stage_d2_generalized_patch_late_crown.json",
    )
    p.add_argument(
        "--out-csv",
        default="prior_stage_d2_generalized_patch_late_crown.csv",
    )

    args = p.parse_args()
    raise SystemExit(main(args))
