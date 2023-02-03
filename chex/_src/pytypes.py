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
"""Type definitions to use for type annotations."""

from typing import Any, Iterable, Mapping, Union
import jax
import jax.numpy as jnp
import numpy as np

# Special types of arrays.
ArrayBatched = jax.interpreters.batching.BatchTracer
ArrayNumpy = np.ndarray
ArraySharded = jax.interpreters.pxla.ShardedDeviceArray
# For instance checking, use `isinstance(x, jax.Array)`.
if hasattr(jax, 'Array'):
  ArrayDevice = jax.Array  # jax >= 0.3.20
elif hasattr(jax.interpreters.xla, '_DeviceArray'):  # 0.2.5 < jax < 0.3.20
  ArrayDevice = jax.interpreters.xla._DeviceArray  # pylint:disable=protected-access
else:  # jax <= 0.2.5
  ArrayDevice = jax.interpreters.xla.DeviceArray

# Generic array type.
Array = Union[ArrayDevice, ArrayNumpy, ArrayBatched, ArraySharded]

# A tree of generic arrays.
ArrayTree = Union[Array, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# Other types.
Scalar = Union[float, int]
Numeric = Union[Array, Scalar]
Shape = jax.core.Shape
PRNGKey = jax.random.KeyArray
PyTreeDef = type(jax.tree_util.tree_structure(None))
if hasattr(jax, 'Device'):
  Device = jax.Device  # jax >= 0.4.3
else:
  Device = jax.lib.xla_extension.Device
ArrayDType = type(jnp.float32)
