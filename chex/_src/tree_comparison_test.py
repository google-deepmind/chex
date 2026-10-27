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
"""Tests structural validation in eager and compiled tree comparisons."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np


class TreeComparisonTest(parameterized.TestCase):

  @parameterized.product(
      assertion=(chex.assert_trees_all_equal, chex.assert_trees_all_close),
      compiled=(False, True),
      mismatch=('keys', 'container', 'extra_leaf', 'missing_leaf', 'none'),
      mismatch_index=(1, 2),
  )
  def test_rejects_different_structures(
      self, assertion, compiled, mismatch, mismatch_index
  ):
    leaf = np.array([1., 2.], dtype=np.float32)
    tree = {'params': {'weight': leaf}}
    different_trees = {
        'keys': {'params': {'bias': leaf}},
        'container': {'params': [leaf]},
        'extra_leaf': {'params': {'weight': leaf, 'z': leaf}},
        'missing_leaf': {'params': {}},
        'none': {'params': {'weight': leaf, 'z': None}},
    }

    def compare(*trees):
      assertion(*trees)
      return trees[0]

    if compiled:
      compare = chex.chexify(jax.jit(compare), async_check=False)
    trees = [tree, tree, tree]
    trees[mismatch_index] = different_trees[mismatch]
    with self.assertRaisesRegex(AssertionError, 'struct'):
      compare(*trees)

  @parameterized.product(
      assertion=(chex.assert_trees_all_equal, chex.assert_trees_all_close),
      compiled=(False, True),
  )
  def test_accepts_matching_structures(self, assertion, compiled):
    def compare(*trees):
      assertion(*trees)
      return trees[0]

    if compiled:
      compare = chex.chexify(jax.jit(compare), async_check=False)
    tree = {
        'params': [np.array([1., 2.], dtype=np.float32), None],
        'empty': np.array([], dtype=np.float32),
    }
    result = compare(tree, tree, tree)
    np.testing.assert_array_equal(result['params'][0], tree['params'][0])
    compare({}, {}, {})

  @parameterized.product(
      assertion=(chex.assert_trees_all_equal, chex.assert_trees_all_close),
      compiled=(False, True),
  )
  def test_rejects_unequal_values(self, assertion, compiled):
    def compare(*trees):
      assertion(*trees)
      return trees[0]

    if compiled:
      compare = chex.chexify(jax.jit(compare), async_check=False)
    tree = {'params': [np.array([1., 2.], dtype=np.float32)]}
    different = {'params': [np.array([1., 4.], dtype=np.float32)]}
    with self.assertRaisesRegex(AssertionError, 'equal'):
      compare(tree, tree, different)

  @parameterized.parameters(
      chex.assert_trees_all_equal, chex.assert_trees_all_close
  )
  def test_structural_error_has_custom_message(self, assertion):
    def compare(x, y):
      assertion(x, y, custom_message='parameter state mismatch')
      return x

    checked = chex.chexify(jax.jit(compare), async_check=False)
    leaf = np.array([1., 2.], dtype=np.float32)
    with self.assertRaisesRegex(AssertionError, 'parameter state mismatch'):
      checked({'weight': leaf}, {'bias': leaf})

  @parameterized.parameters(
      chex.assert_trees_all_equal, chex.assert_trees_all_close
  )
  def test_structural_assertion_can_be_disabled(self, assertion):
    def compare(x, y):
      assertion(x, y)
      return x

    checked = chex.chexify(jax.jit(compare), async_check=False)
    leaf = np.array([1., 2.], dtype=np.float32)
    chex.disable_asserts()
    try:
      result = checked({'weight': leaf}, {'bias': leaf})
    finally:
      chex.enable_asserts()
    np.testing.assert_array_equal(result['weight'], leaf)


if __name__ == '__main__':
  absltest.main()
