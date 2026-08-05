"""Algebra kernels — single source of truth.

The frozen reference module ``reference/structure_tensor.py`` is normative
(PROTOCOL_v3.md section 12). Rather than re-implement (and risk divergence),
production code imports the reference functions directly. Bit-exactness tests
(section 13) then compare the sparse structure-tensor contraction, the integer
reference, and the signed-permutation kernel against each other.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from ..paths import REPO_ROOT

_ref_path = REPO_ROOT / "reference" / "structure_tensor.py"
_spec = importlib.util.spec_from_file_location("sextant._frozen_structure_tensor", _ref_path)
assert _spec and _spec.loader
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

# Re-export the frozen names.
NCOMP = _ref.NCOMP
QMAX_ACT = _ref.QMAX_ACT
cocycle_sign = _ref.cocycle_sign
build_structure_tensor = _ref.build_structure_tensor
ternarize_weight_global_ste = _ref.ternarize_weight_global_ste
quantize_activation_per_token_ste = _ref.quantize_activation_per_token_ste
algebra_linear_training = _ref.algebra_linear_training
algebra_linear_integer_reference = _ref.algebra_linear_integer_reference
algebra_linear_signed_permutation_reference = _ref.algebra_linear_signed_permutation_reference

__all__ = [
    "NCOMP",
    "QMAX_ACT",
    "cocycle_sign",
    "build_structure_tensor",
    "ternarize_weight_global_ste",
    "quantize_activation_per_token_ste",
    "algebra_linear_training",
    "algebra_linear_integer_reference",
    "algebra_linear_signed_permutation_reference",
]
