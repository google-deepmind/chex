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
"""Scalar bfloat16 precision checks must behave like single-element arrays."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np


class ScalarBfloatUlpTest(parameterized.TestCase):

  @parameterized.product(
      value=[-2.0, -0.0, 0.0, 0.5, 1.0, 100.0], host=[False, True]
  )
  def test_identical_scalar_values_pass_zero_tolerance(self, value, host):
    a = jnp.asarray(value, jnp.bfloat16)
    if host:
      a = np.asarray(a)
    chex.assert_trees_all_close_ulp(a, a, maxulp=0)

  @parameterized.product(value=[-2.0, 0.5, 1.0, 100.0], host=[False, True])
  def test_adjacent_values_pass_one_ulp_and_fail_zero_ulp(self, value, host):
    a = jnp.asarray(value, jnp.bfloat16)
    b = jnp.nextafter(a, jnp.asarray(jnp.inf, jnp.bfloat16))
    if host:
      a, b = np.asarray(a), np.asarray(b)
    chex.assert_trees_all_close_ulp({'stat': a}, {'stat': b}, maxulp=1)
    with self.assertRaises(AssertionError):
      chex.assert_trees_all_close_ulp({'stat': a}, {'stat': b}, maxulp=0)

  def test_two_steps_are_rejected_without_a_dtype_view_exception(self):
    a = jnp.asarray(1.0, jnp.bfloat16)
    b = jnp.nextafter(
        jnp.nextafter(a, jnp.asarray(2.0, a.dtype)), jnp.asarray(2.0, a.dtype)
    )
    with self.assertRaises(AssertionError):
      chex.assert_trees_all_close_ulp(a, b, maxulp=1)
    chex.assert_trees_all_close_ulp(a, b, maxulp=2)

  def test_jitted_scalar_outputs_can_be_checked_on_the_host(self):
    calculate = jax.jit(lambda x: jnp.mean(x))
    actual = calculate(jnp.ones((8,), jnp.bfloat16))
    chex.assert_trees_all_close_ulp(
        actual, jnp.asarray(1.0, jnp.bfloat16), maxulp=0
    )

  def test_array_and_other_floating_dtype_controls_are_unchanged(self):
    for shape in [(1,), (2, 3)]:
      a = jnp.ones(shape, jnp.bfloat16)
      b = jnp.nextafter(a, jnp.full_like(a, 2.0))
      chex.assert_trees_all_close_ulp(a, b, maxulp=1)
      with self.assertRaises(AssertionError):
        chex.assert_trees_all_close_ulp(a, b, maxulp=0)
    for dtype in [jnp.float16, jnp.float32]:
      a = jnp.asarray(1.0, dtype)
      chex.assert_trees_all_close_ulp(a, a, maxulp=0)


if __name__ == '__main__':
  absltest.main()
