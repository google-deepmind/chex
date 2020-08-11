# Lint as: python3
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

from chex._src.asserts import assert_devices_available
from chex._src.asserts import assert_equal_shape
from chex._src.asserts import assert_gpu_available
from chex._src.asserts import assert_numerical_grads
from chex._src.asserts import assert_rank
from chex._src.asserts import assert_scalar
from chex._src.asserts import assert_scalar_in
from chex._src.asserts import assert_scalar_negative
from chex._src.asserts import assert_scalar_non_negative
from chex._src.asserts import assert_scalar_positive
from chex._src.asserts import assert_shape
from chex._src.asserts import assert_tpu_available
from chex._src.asserts import assert_tree_all_close
from chex._src.asserts import assert_tree_all_finite
from chex._src.asserts import assert_type
from chex._src.dataclass import dataclass
from chex._src.fake import fake_jit
from chex._src.fake import fake_pmap
from chex._src.fake import fake_pmap_and_jit
from chex._src.fake import set_n_cpu_devices
from chex._src.pytypes import Array
from chex._src.pytypes import ArrayBatched
from chex._src.pytypes import ArrayDevice
from chex._src.pytypes import ArrayNumpy
from chex._src.pytypes import Numeric
from chex._src.pytypes import PRNGKey
from chex._src.pytypes import Scalar
from chex._src.variants import all_variants
from chex._src.variants import TestCase
from chex._src.variants import variants

__version__ = "0.0.2"

__all__ = (
    "all_variants",
    "Array",
    "ArrayBatched",
    "ArrayDevice",
    "ArrayNumpy",
    "assert_devices_available",
    "assert_equal_shape",
    "assert_numerical_grads",
    "assert_scalar",
    "assert_scalar_in",
    "assert_scalar_negative",
    "assert_scalar_non_negative",
    "assert_scalar_positive",
    "assert_shape",
    "assert_tpu_available",
    "assert_tree_all_close",
    "assert_tree_all_finite",
    "assert_type",
    "dataclass",
    "fake_jit",
    "fake_pmap",
    "fake_pmap_and_jit",
    "Numeric",
    "PRNGKey",
    "assert_rank",
    "Scalar",
    "set_n_cpu_devices",
    "TestCase",
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
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
