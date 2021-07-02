# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Chex: Testing made fun, in JAX!"""

from chex._src.asserts import assert_axis_dimension
from chex._src.asserts import assert_axis_dimension_gt
from chex._src.asserts import assert_devices_available
from chex._src.asserts import assert_equal
from chex._src.asserts import assert_equal_rank
from chex._src.asserts import assert_equal_shape
from chex._src.asserts import assert_equal_shape_prefix
from chex._src.asserts import assert_equal_shape_suffix
from chex._src.asserts import assert_exactly_one_is_none
from chex._src.asserts import assert_gpu_available
from chex._src.asserts import assert_is_broadcastable
from chex._src.asserts import assert_is_divisible
from chex._src.asserts import assert_max_traces
from chex._src.asserts import assert_not_both_none
from chex._src.asserts import assert_numerical_grads
from chex._src.asserts import assert_rank
from chex._src.asserts import assert_scalar
from chex._src.asserts import assert_scalar_in
from chex._src.asserts import assert_scalar_negative
from chex._src.asserts import assert_scalar_non_negative
from chex._src.asserts import assert_scalar_positive
from chex._src.asserts import assert_shape
from chex._src.asserts import assert_tpu_available
from chex._src.asserts import assert_tree_all_close  # Deprecated
from chex._src.asserts import assert_tree_all_equal_comparator  # Deprecated
from chex._src.asserts import assert_tree_all_equal_shapes  # Deprecated
from chex._src.asserts import assert_tree_all_equal_structs  # Deprecated
from chex._src.asserts import assert_tree_all_finite
from chex._src.asserts import assert_tree_no_nones
from chex._src.asserts import assert_tree_shape_prefix
from chex._src.asserts import assert_trees_all_close
from chex._src.asserts import assert_trees_all_equal_comparator
from chex._src.asserts import assert_trees_all_equal_shapes
from chex._src.asserts import assert_trees_all_equal_structs
from chex._src.asserts import assert_type
from chex._src.asserts import clear_trace_counter
from chex._src.asserts import if_args_not_none
from chex._src.dataclass import dataclass
from chex._src.dataclass import mappable_dataclass
from chex._src.fake import fake_jit
from chex._src.fake import fake_pmap
from chex._src.fake import fake_pmap_and_jit
from chex._src.fake import set_n_cpu_devices
from chex._src.pytypes import Array
from chex._src.pytypes import ArrayBatched
from chex._src.pytypes import ArrayDevice
from chex._src.pytypes import ArrayNumpy
from chex._src.pytypes import ArraySharded
from chex._src.pytypes import ArrayTree
from chex._src.pytypes import CpuDevice
from chex._src.pytypes import Device
from chex._src.pytypes import GpuDevice
from chex._src.pytypes import Numeric
from chex._src.pytypes import PRNGKey
from chex._src.pytypes import Scalar
from chex._src.pytypes import Shape
from chex._src.pytypes import TpuDevice
from chex._src.variants import all_variants
from chex._src.variants import ChexVariantType
from chex._src.variants import params_product
from chex._src.variants import TestCase
from chex._src.variants import variants


__version__ = "0.0.7"

__all__ = (
    "all_variants",
    "Array",
    "ArrayBatched",
    "ArrayDevice",
    "ArrayNumpy",
    "ArraySharded",
    "ArrayTree",
    "assert_axis_dimension",
    "assert_axis_dimension_gt",
    "assert_devices_available",
    "assert_equal",
    "assert_equal_rank",
    "assert_equal_shape",
    "assert_equal_shape_prefix",
    "assert_equal_shape_suffix",
    "assert_exactly_one_is_none",
    "assert_gpu_available",
    "assert_is_broadcastable",
    "assert_is_divisible",
    "assert_max_traces",
    "assert_not_both_none",
    "assert_numerical_grads",
    "assert_rank",
    "assert_scalar",
    "assert_scalar_in",
    "assert_scalar_negative",
    "assert_scalar_non_negative",
    "assert_scalar_positive",
    "assert_shape",
    "assert_tpu_available",
    "assert_tree_all_close",  # Deprecated
    "assert_tree_all_equal_comparator",  # Deprecated
    "assert_tree_all_equal_shapes",  # Deprecated
    "assert_tree_all_equal_structs",  # Deprecated
    "assert_tree_all_finite",
    "assert_tree_no_nones",
    "assert_tree_shape_prefix",
    "assert_trees_all_close",
    "assert_trees_all_equal_comparator",
    "assert_trees_all_equal_shapes",
    "assert_trees_all_equal_structs",
    "assert_type",
    "ChexVariantType",
    "clear_trace_counter",
    "CpuDevice",
    "dataclass",
    "Device",
    "fake_jit",
    "fake_pmap",
    "fake_pmap_and_jit",
    "GpuDevice",
    "if_args_not_none",
    "mappable_dataclass",
    "Numeric",
    "params_product",
    "PRNGKey",
    "Scalar",
    "set_n_cpu_devices",
    "Shape",
    "TestCase",
    "TpuDevice",
    "variants",
)


#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Chex public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
