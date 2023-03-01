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

from typing import Any, TypeAlias, Union
import jax
import jax.numpy as jnp
import numpy as np

# Special types of arrays.
ArrayBatched: TypeAlias = jax.interpreters.batching.BatchTracer
ArrayNumpy: TypeAlias = np.ndarray
ArraySharded: TypeAlias = jax.interpreters.pxla.ShardedDeviceArray
# For instance checking, use `isinstance(x, jax.Array)`.
ArrayDevice: TypeAlias = jax.Array  # jax >= 0.3.20

# Generic array type.
Array = Union[jax.Array, np.ndarray]
ArrayLike: TypeAlias = jax.typing.ArrayLike

# A tree of generic arrays.
ArrayTree = Any
# Union[Array, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# Other types.
Scalar = Union[float, int]
Numeric = Union[Array, Scalar]
Shape: TypeAlias = jax.core.Shape
PRNGKey: TypeAlias = jax.random.KeyArray
PyTreeDef: TypeAlias = jax.tree_util.PyTreeDef
Device: TypeAlias = jax.Device  # jax >= 0.4.3
ArrayDType = type(jnp.float32)
