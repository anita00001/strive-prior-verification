#!/usr/bin/env python3
"""
STRIVE Stage D2 non-generalized verifier.

Definition
----------
Perturb exactly ONE spatial location in the selected target agent's local map,
jointly across all four raster channels:

    1 x 1 x 4 = 4 uncertain values.

The verifier reuses the same sound custom-normalization backward-CROWN engine
as verify_prior_stage_d2_generalized.py.  It uses the same unified
prior_stage_d_capture.pt and supports any number of incoming target edges.

Everything except those four raster values is held fixed.

Default property:
    epsilon = 1e-4
    delta   = 0.05

NOT_PROVED means the verifier did not establish the property; it is not by
itself a counterexample.
"""

from verify_prior_stage_d2_generalized import make_parser, main


if __name__ == "__main__":
    parser = make_parser(
        default_patch_size=1,
        default_out_prefix="prior_stage_d2",
    )
    args = parser.parse_args()

    # Non-generalized D2 is deliberately fixed to one spatial raster location.
    if int(args.patch_size) != 1:
        parser.error(
            "verify_prior_stage_d2.py defines non-generalized D2 as "
            "patch-size 1 (4 channels = 4 uncertain values)."
        )

    raise SystemExit(
        main(
            args,
            mode="non-generalized",
        )
    )
