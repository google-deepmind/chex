# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for ``dimensions`` module."""

import doctest

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import dimensions
import jax
import numpy as np


class _ChexModule:
  """Mock module for providing minimal context to docstring tests."""
  assert_shape = asserts.assert_shape
  assert_rank = asserts.assert_rank
  Dimensions = dimensions.Dimensions  # pylint: disable=invalid-name


class DimensionsTest(parameterized.TestCase):

  def test_docstring_examples(self):
    doctest.run_docstring_examples(
        dimensions.Dimensions,
        globs={'chex': _ChexModule, 'jax': jax, 'jnp': jax.numpy})

  @parameterized.named_parameters([
      ('scalar', '', (), ()),
      ('vector', 'a', (7,), (7,)),
      ('list', 'ab', [7, 11], (7, 11)),
      ('numpy_array', 'abc', np.array([7, 11, 13]), (7, 11, 13)),
      ('case_sensitive', 'aA', (7, 11), (7, 11)),
  ])
  def test_set_ok(self, k, v, shape):
    dims = dimensions.Dimensions(x=23, y=29)
    dims[k] = v
    asserts.assert_shape(np.empty((23, *shape, 29)), dims['x' + k + 'y'])

  def test_set_wildcard(self):
    dims = dimensions.Dimensions(x=23, y=29)
    dims['a_b__'] = (7, 11, 13, 17, 19)
    self.assertEqual(dims['xayb'], (23, 7, 29, 13))
    with self.assertRaisesRegex(KeyError, r'\*'):
      dims['ab*'] = (7, 11, 13)

  def test_get_wildcard(self):
    dims = dimensions.Dimensions(x=23, y=29)
    self.assertEqual(dims['x*y**'], (23, None, 29, None, None))
    asserts.assert_shape(np.empty((23, 1, 29, 2, 3)), dims['x*y**'])
    with self.assertRaisesRegex(KeyError, r'\_'):
      dims['xy_']  # pylint: disable=pointless-statement

  def test_get_literals(self):
    dims = dimensions.Dimensions(x=23, y=29)
    self.assertEqual(dims['x1y23'], (23, 1, 29, 2, 3))

  @parameterized.named_parameters([
      ('scalar', 'a', 7, TypeError, r'value must be sized'),
      ('iterator', 'a', (x for x in [7]), TypeError, r'value must be sized'),
      ('len_mismatch', 'ab', (7, 11, 13), ValueError, r'different length'),
      ('non_integer_size', 'a', (7.001,),
       TypeError, r'cannot be interpreted as a python int'),
      ('bad_key_type', 13, (7,), TypeError, r'key must be a string'),
      ('bad_key_string', '@%^#', (7, 11, 13, 17), KeyError, r'\@'),
  ])
  def test_set_exception(self, k, v, e, m):
    dims = dimensions.Dimensions(x=23, y=29)
    with self.assertRaisesRegex(e, m):
      dims[k] = v

  @parameterized.named_parameters([
      ('bad_key_type', 13, TypeError, r'key must be a string'),
      ('bad_key_string', '@%^#', KeyError, r'\@'),
  ])
  def test_get_exception(self, k, e, m):
    dims = dimensions.Dimensions(x=23, y=29)
    with self.assertRaisesRegex(e, m):
      dims[k]  # pylint: disable=pointless-statement


if __name__ == '__main__':
  absltest.main()
