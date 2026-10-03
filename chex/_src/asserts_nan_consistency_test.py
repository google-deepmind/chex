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
"""Matching-NaN consistency across eager and compiled tree assertions."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np


class AllCloseNanConsistencyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("scalar_nan", [np.nan], [np.nan], True),
      ("matching_nans", [np.nan, 1.0, 2.0], [np.nan, 1.0, 2.0], True),
      ("finite_tolerance", [np.nan, 1.0], [np.nan, 1.001], True),
      ("different_nan_positions", [np.nan, 1.0], [1.0, np.nan], False),
      ("nan_vs_finite", [np.nan], [0.0], False),
      ("finite_vs_nan", [0.0], [np.nan], False),
      ("finite_mismatch", [np.nan, 1.0], [np.nan, 3.0], False),
      ("matching_infinities", [np.nan, np.inf], [np.nan, np.inf], True),
      ("different_infinities", [np.inf], [-np.inf], False),
      ("empty", [], [], True),
  )
  def test_eager_and_compiled_results_agree(self, first, second, matches):
    first, second = jnp.asarray(first), jnp.asarray(second)

    def compare(a, b):
      chex.assert_trees_all_close(a, b, rtol=0.01, atol=0.001)
      return jnp.zeros((), dtype=jnp.int32)

    checked = chex.chexify(jax.jit(compare), async_check=False)
    if matches:
      self.assertEqual(compare(first, second), 0)
      self.assertEqual(checked(first, second), 0)
    else:
      for call in (compare, checked):
        with (
            self.subTest(call=call),
            self.assertRaisesRegex(AssertionError, "not approximately equal"),
        ):
          call(first, second)
    checked.wait_checks()

  def test_three_nested_trees_and_custom_diagnostics(self):
    def compare(first, second, third):
      chex.assert_trees_all_close(
          first, second, third, custom_message="masked measurements"
      )
      return jnp.asarray(1)

    checked = chex.chexify(jax.jit(compare), async_check=False)
    tree = {
        "values": (jnp.array([np.nan, 2.0]),),
        "scalar": jnp.asarray(np.nan),
    }
    self.assertEqual(checked(tree, tree, tree), 1)
    bad = {"values": (jnp.array([np.nan, 4.0]),), "scalar": jnp.asarray(np.nan)}
    with self.assertRaisesRegex(AssertionError, "masked measurements"):
      checked(tree, tree, bad)
    checked.wait_checks()

  def test_masked_batch_comparisons_work_under_vmap(self):
    def compare(row, mask, expected):
      actual = jnp.where(mask, row * 2.0, jnp.nan)
      chex.assert_trees_all_close(actual, expected)
      return actual

    checked = chex.chexify(jax.jit(jax.vmap(compare)), async_check=False)
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([[True, False], [False, True]])
    expected = jnp.array([[2.0, np.nan], [np.nan, 8.0]])
    np.testing.assert_allclose(checked(values, mask, expected), expected)
    checked.wait_checks()


if __name__ == "__main__":
  absltest.main()
