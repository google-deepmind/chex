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
"""Tests for `dataclasses.py`."""

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import dataclass
from chex._src import pytypes
import jax
import numpy as np


@dataclass.dataclass
class NestedDataclass():
  c: pytypes.ArrayDevice
  d: pytypes.ArrayDevice


@dataclass.dataclass
class Dataclass():
  a: NestedDataclass
  b: pytypes.ArrayDevice


@dataclass.dataclass(frozen=True)
class FrozenDataclass():
  a: NestedDataclass
  b: pytypes.ArrayDevice


def dummy_dataclass(factor=1., frozen=False):
  class_ctor = FrozenDataclass if frozen else Dataclass
  return class_ctor(
      a=NestedDataclass(
          c=factor * np.ones((3,), dtype=np.float32),
          d=factor * np.ones((4,), dtype=np.float32)),
      b=factor * 2 * np.ones((5,), dtype=np.float32))


class DataclassesTest(parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_dataclass_tree_leaves(self, frozen):
    obj = dummy_dataclass(frozen=frozen)
    self.assertLen(jax.tree_util.tree_leaves(obj), 3)

  @parameterized.parameters([True, False])
  def test_dataclass_tree_map(self, frozen):
    factor = 5.
    obj = dummy_dataclass(frozen=frozen)
    target_obj = dummy_dataclass(factor=factor, frozen=frozen)
    asserts.assert_tree_all_close(
        jax.tree_util.tree_map(lambda t: factor * t, obj), target_obj)

  @parameterized.parameters([True, False])
  def test_dataclass_replace(self, frozen):
    factor = 5.
    obj = dummy_dataclass(frozen=frozen)
    obj = obj.replace(a=obj.a.replace(c=factor * obj.a.c))
    obj = obj.replace(a=obj.a.replace(d=factor * obj.a.d))
    obj = obj.replace(b=factor * obj.b)
    target_obj = dummy_dataclass(factor=factor, frozen=frozen)
    asserts.assert_tree_all_close(obj, target_obj)

  def test_unfrozen_dataclass_is_mutable(self):
    factor = 5.
    obj = dummy_dataclass(frozen=False)
    obj.a.c = factor * obj.a.c
    obj.a.d = factor * obj.a.d
    obj.b = factor * obj.b
    target_obj = dummy_dataclass(factor=factor, frozen=False)
    asserts.assert_tree_all_close(obj, target_obj)

  def test_frozen_dataclass_raise_error(self):
    factor = 5.
    obj = dummy_dataclass(frozen=True)
    obj.a.c = factor * obj.a.c  # mutable since obj.a is not frozen.
    with self.assertRaisesRegex(
        dataclass.FrozenInstanceError, 'cannot assign to field'):
      obj.b = factor * obj.b  # raises error because obj is frozen.

if __name__ == '__main__':
  absltest.main()
