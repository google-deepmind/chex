# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Shape consistency for eager and compiled close assertions."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np


class AssertTreesAllCloseShapesTest(parameterized.TestCase):

  @parameterized.product(
      shapes=(((2, 1), (2,)), ((1,), (3,)), ((0, 1), (0, 3))),
      reverse=(False, True),
      compiled=(False, True),
      mismatch_index=(1, 2),
  )
  def test_rejects_broadcastable_non_scalar_shapes(
      self, shapes, reverse, compiled, mismatch_index
  ):
    def compare(*trees):
      chex.assert_trees_all_close(*trees, custom_message='prediction shape')
      return trees[0]

    call = compare
    if compiled:
      call = chex.chexify(jax.jit(compare), async_check=False)
    if reverse:
      shapes = shapes[::-1]
    trees = [{'prediction': jnp.ones(shapes[0])} for _ in range(3)]
    trees[mismatch_index] = {'prediction': jnp.ones(shapes[1])}
    try:
      with self.assertRaisesRegex(AssertionError, 'prediction shape'):
        call(*trees)
    finally:
      if compiled:
        call.wait_checks()

  @parameterized.product(
      shapes=(((), (2, 3)), ((), (0, 2)), ((), ()), ((2, 3), (2, 3))),
      reverse=(False, True),
      async_check=(False, True),
  )
  def test_accepts_equal_shapes_and_scalar_broadcasting(
      self, shapes, reverse, async_check
  ):
    def compare(x, y):
      chex.assert_trees_all_close(x, y, atol=0.01)
      return x

    if reverse:
      shapes = shapes[::-1]
    x, y = jnp.ones(shapes[0]), jnp.full(shapes[1], 1.001)
    compare(x, y)
    checked = chex.chexify(jax.jit(compare), async_check=async_check)
    np.testing.assert_array_equal(checked(x, y), x)
    checked.wait_checks()

  @parameterized.parameters(False, True)
  def test_rejects_different_values(self, async_check):
    def compare(x, y):
      chex.assert_trees_all_close(x, y)
      return x

    checked = chex.chexify(jax.jit(compare), async_check=async_check)
    with self.assertRaisesRegex(AssertionError, 'not approximately equal'):
      checked(jnp.ones((2, 3)), jnp.zeros((2, 3)))
      checked.wait_checks()
    checked.wait_checks()


if __name__ == '__main__':
  absltest.main()
