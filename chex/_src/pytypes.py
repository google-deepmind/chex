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
"""Pytypes for arrays and scalars."""

from typing import Any, Iterable, Mapping, Union
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
ArrayBatched = jax.interpreters.batching.BatchTracer
ArrayDevice = jax.interpreters.xla.DeviceArray
ArrayNumpy = np.ndarray
ArraySharded = jax.interpreters.pxla.ShardedDeviceArray

Scalar = Union[float, int]
Numeric = Union[Array, Scalar]
PRNGKey = Array

# As of 06/2020 pytype doesn't support recursive types (see b/109648354)
# pytype: disable=not-supported-yet
ArrayTree = Union[Array, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]
