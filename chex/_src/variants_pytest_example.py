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
"""Example of using `chex.variants` with `pytest`."""

from typing import Callable
import chex
from chex._src import variants
import jax.numpy as jnp
import pytest

# `chex.variants` is primarily designed for `unittest.TestCase` and `absl.testing`.
# When using `pytest`, you can manually parametrize your tests over
# `variants.ALL_VARIANTS` (or a subset thereof) to achieve similar coverage.
#
# Note: `chex.variants` manages different JAX execution modes (JIT, PMAP, etc.)
# by decorating the function-under-test.


@pytest.mark.parametrize("variant", variants.ALL_VARIANTS)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_variants_with_pytest(variant: Callable, n: int) -> None:
  """Tests a function across all Chex variants using pytest parametrization.

  Args:
    variant: A Chex variant decorator (e.g., with_jit, without_jit, etc.).
    n: Input parameter for the test.
  """

  # Define the computation you want to test.
  # The `@variant` decorator will apply the specific execution mode
  # (e.g., wrap in jax.jit, jax.pmap) for this test iteration.
  @variant
  def computation(x):
    return x * x

  # Execute the decorated function.
  # Convert input to JAX array as some variants (like pmap) might expect it
  # or handle it differently.
  result = computation(jnp.array(n))

  # Verify the result.
  assert result == n * n
