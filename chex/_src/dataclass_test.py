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
"""Tests for `dataclass.py`."""
import copy
import dataclasses
import sys
from typing import Any, Mapping, TypeVar, Generic

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import dataclass
from chex._src import pytypes
import jax
import numpy as np
import tree

chex_dataclass = dataclass.dataclass
mappable_dataclass = dataclass.mappable_dataclass
orig_dataclass = dataclasses.dataclass


@chex_dataclass
class NestedDataclass():
  c: pytypes.ArrayDevice
  d: pytypes.ArrayDevice


@chex_dataclass
class Dataclass():
  a: NestedDataclass
  b: pytypes.ArrayDevice


@chex_dataclass(frozen=True)
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


@orig_dataclass
class ClassWithoutMap:
  k: dict  # pylint:disable=g-bare-generic

  def some_method(self, *args):
    raise RuntimeError('ClassWithoutMap.some_method() was called.')


def _get_mappable_dataclasses(test_type):
  """Generates shallow and nested mappable dataclasses."""

  class Class:
    """Shallow class."""

    k_tuple: tuple  # pylint:disable=g-bare-generic
    k_dict: dict  # pylint:disable=g-bare-generic

    def some_method(self, *args):
      raise RuntimeError('Class.some_method() was called.')

  class NestedClass:
    """Nested class."""

    k_any: Any
    k_int: int
    k_str: str
    k_arr: np.ndarray
    k_dclass_with_map: Class
    k_dclass_no_map: ClassWithoutMap
    k_dict_factory: dict = dataclasses.field(  # pylint:disable=g-bare-generic
        default_factory=lambda: dict(x='x', y='y'))
    k_default: str = 'default_str'
    k_non_init: int = dataclasses.field(default=1, init=False)
    k_init_only: dataclasses.InitVar[int] = 10

    def some_method(self, *args):
      raise RuntimeError('NestedClassWithMap.some_method() was called.')

    def __post_init__(self, k_init_only):
      self.k_non_init = self.k_int * k_init_only

  if test_type == 'chex':
    cls = chex_dataclass(Class, mappable_dataclass=True)
    nested_cls = chex_dataclass(NestedClass, mappable_dataclass=True)
  elif test_type == 'original':
    cls = mappable_dataclass(orig_dataclass(Class))
    nested_cls = mappable_dataclass(orig_dataclass(NestedClass))
  else:
    raise ValueError(f'Unknown test type: {test_type}')

  return cls, nested_cls


@parameterized.named_parameters(('_original', 'original'), ('_chex', 'chex'))
class MappableDataclassTest(parameterized.TestCase):

  def _init_testdata(self, test_type):
    """Initializes test data."""
    map_cls, nested_map_cls = _get_mappable_dataclasses(test_type)

    self.dcls_with_map_inner = map_cls(
        k_tuple=(1, 2), k_dict=dict(k1=32, k2=33))
    self.dcls_with_map_inner_inc = map_cls(
        k_tuple=(2, 3), k_dict=dict(k1=33, k2=34))

    self.dcls_no_map = ClassWithoutMap(k=dict(t='t', t2='t2'))
    self.dcls_with_map = nested_map_cls(
        k_any=None,
        k_int=1,
        k_str='test_str',
        k_arr=np.array(16),
        k_dclass_with_map=self.dcls_with_map_inner,
        k_dclass_no_map=self.dcls_no_map)

    self.dcls_with_map_inc_ints = nested_map_cls(
        k_any=None,
        k_int=2,
        k_str='test_str',
        k_arr=np.array(16),
        k_dclass_with_map=self.dcls_with_map_inner_inc,
        k_dclass_no_map=self.dcls_no_map,
        k_default='default_str')

    self.dcls_flattened_with_path = [
        (('k_any',), None),
        (('k_arr',), np.array(16)),
        (('k_dclass_no_map',), self.dcls_no_map),
        (('k_dclass_with_map', 'k_dict', 'k1'), 32),
        (('k_dclass_with_map', 'k_dict', 'k2'), 33),
        (('k_dclass_with_map', 'k_tuple', 0), 1),
        (('k_dclass_with_map', 'k_tuple', 1), 2),
        (('k_default',), 'default_str'),
        (('k_dict_factory', 'x'), 'x'),
        (('k_dict_factory', 'y'), 'y'),
        (('k_int',), 1),
        (('k_non_init',), 10),
        (('k_str',), 'test_str'),
    ]

    self.dcls_flattened_with_path_up_to = [
        (('k_any',), None),
        (('k_arr',), np.array(16)),
        (('k_dclass_no_map',), self.dcls_no_map),
        (('k_dclass_with_map',), self.dcls_with_map_inner),
        (('k_default',), 'default_str'),
        (('k_dict_factory', 'x'), 'x'),
        (('k_dict_factory', 'y'), 'y'),
        (('k_int',), 1),
        (('k_non_init',), 10),
        (('k_str',), 'test_str'),
    ]

    self.dcls_flattened = [v for (_, v) in self.dcls_flattened_with_path]
    self.dcls_flattened_up_to = [
        v for (_, v) in self.dcls_flattened_with_path_up_to
    ]
    self.dcls_tree_size = 18
    self.dcls_tree_size_no_dicts = 14

  def testFlattenAndUnflatten(self, test_type):
    self._init_testdata(test_type)

    self.assertEqual(self.dcls_flattened, tree.flatten(self.dcls_with_map))
    self.assertEqual(
        self.dcls_with_map,
        tree.unflatten_as(self.dcls_with_map_inc_ints, self.dcls_flattened))

    dataclass_in_seq = [34, self.dcls_with_map, [1, 2]]
    dataclass_in_seq_flat = [34] + self.dcls_flattened + [1, 2]
    self.assertEqual(dataclass_in_seq_flat, tree.flatten(dataclass_in_seq))
    self.assertEqual(dataclass_in_seq,
                     tree.unflatten_as(dataclass_in_seq, dataclass_in_seq_flat))

  def testFlattenUpTo(self, test_type):
    self._init_testdata(test_type)
    structure = copy.copy(self.dcls_with_map)
    structure.k_dclass_with_map = None  # Do not flatten 'k_dclass_with_map'
    self.assertEqual(self.dcls_flattened_up_to,
                     tree.flatten_up_to(structure, self.dcls_with_map))

  def testFlattenWithPath(self, test_type):
    self._init_testdata(test_type)

    self.assertEqual(
        tree.flatten_with_path(self.dcls_with_map),
        self.dcls_flattened_with_path)

  def testFlattenWithPathUpTo(self, test_type):
    self._init_testdata(test_type)
    structure = copy.copy(self.dcls_with_map)
    structure.k_dclass_with_map = None  # Do not flatten 'k_dclass_with_map'
    self.assertEqual(
        tree.flatten_with_path_up_to(structure, self.dcls_with_map),
        self.dcls_flattened_with_path_up_to)

  def testMapStructure(self, test_type):
    self._init_testdata(test_type)

    add_one_to_ints_fn = lambda x: x + 1 if isinstance(x, int) else x
    mapped_inc_ints = tree.map_structure(add_one_to_ints_fn, self.dcls_with_map)

    self.assertEqual(self.dcls_with_map_inc_ints, mapped_inc_ints)
    self.assertEqual(self.dcls_with_map_inc_ints.k_non_init,
                     self.dcls_with_map_inc_ints.k_int * 10)
    self.assertEqual(mapped_inc_ints.k_non_init, mapped_inc_ints.k_int * 10)

  def testMapStructureUpTo(self, test_type):
    self._init_testdata(test_type)

    structure = copy.copy(self.dcls_with_map)
    structure.k_dclass_with_map = None  # Do not map over 'k_dclass_with_map'
    add_one_to_ints_fn = lambda x: x + 1 if isinstance(x, int) else x
    mapped_inc_ints = tree.map_structure_up_to(structure, add_one_to_ints_fn,
                                               self.dcls_with_map)

    # k_dclass_with_map should be passed through unchanged
    class_with_map = self.dcls_with_map.k_dclass_with_map
    self.dcls_with_map_inc_ints.k_dclass_with_map = class_with_map
    self.assertEqual(self.dcls_with_map_inc_ints, mapped_inc_ints)
    self.assertEqual(self.dcls_with_map_inc_ints.k_non_init,
                     self.dcls_with_map_inc_ints.k_int * 10)
    self.assertEqual(mapped_inc_ints.k_non_init, mapped_inc_ints.k_int * 10)

  def testMapStructureWithPath(self, test_type):
    self._init_testdata(test_type)

    add_one_to_ints_fn = lambda path, x: x + 1 if isinstance(x, int) else x
    mapped_inc_ints = tree.map_structure_with_path(add_one_to_ints_fn,
                                                   self.dcls_with_map)

    self.assertEqual(self.dcls_with_map_inc_ints, mapped_inc_ints)
    self.assertEqual(self.dcls_with_map_inc_ints.k_non_init,
                     self.dcls_with_map_inc_ints.k_int * 10)
    self.assertEqual(mapped_inc_ints.k_non_init, mapped_inc_ints.k_int * 10)

  def testMapStructureWithPathUpTo(self, test_type):
    self._init_testdata(test_type)

    structure = copy.copy(self.dcls_with_map)
    structure.k_dclass_with_map = None  # Do not map over 'k_dclass_with_map'
    add_one_to_ints_fn = lambda path, x: x + 1 if isinstance(x, int) else x
    mapped_inc_ints = tree.map_structure_with_path_up_to(
        structure, add_one_to_ints_fn, self.dcls_with_map)

    # k_dclass_with_map should be passed through unchanged
    class_with_map = self.dcls_with_map.k_dclass_with_map
    self.dcls_with_map_inc_ints.k_dclass_with_map = class_with_map

    self.assertEqual(self.dcls_with_map_inc_ints, mapped_inc_ints)
    self.assertEqual(self.dcls_with_map_inc_ints.k_non_init,
                     self.dcls_with_map_inc_ints.k_int * 10)
    self.assertEqual(mapped_inc_ints.k_non_init, mapped_inc_ints.k_int * 10)

  def testTraverse(self, test_type):
    self._init_testdata(test_type)

    visited = []
    tree.traverse(visited.append, self.dcls_with_map, top_down=False)
    self.assertLen(visited, self.dcls_tree_size)

    visited_without_dicts = []

    def visit_without_dicts(x):
      visited_without_dicts.append(x)
      return 'X' if isinstance(x, dict) else None

    tree.traverse(visit_without_dicts, self.dcls_with_map, top_down=True)
    self.assertLen(visited_without_dicts, self.dcls_tree_size_no_dicts)

  def testIsDataclass(self, test_type):
    self._init_testdata(test_type)

    self.assertTrue(dataclasses.is_dataclass(self.dcls_no_map))
    self.assertTrue(dataclasses.is_dataclass(self.dcls_with_map))
    self.assertTrue(
        dataclasses.is_dataclass(self.dcls_with_map.k_dclass_with_map))
    self.assertTrue(
        dataclasses.is_dataclass(self.dcls_with_map.k_dclass_no_map))


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
    asserts.assert_trees_all_close(
        jax.tree_util.tree_map(lambda t: factor * t, obj), target_obj)

  @parameterized.parameters([True, False])
  def test_dataclass_replace(self, frozen):
    factor = 5.
    obj = dummy_dataclass(frozen=frozen)
    obj = obj.replace(a=obj.a.replace(c=factor * obj.a.c))
    obj = obj.replace(a=obj.a.replace(d=factor * obj.a.d))
    obj = obj.replace(b=factor * obj.b)
    target_obj = dummy_dataclass(factor=factor, frozen=frozen)
    asserts.assert_trees_all_close(obj, target_obj)

  def test_unfrozen_dataclass_is_mutable(self):
    factor = 5.
    obj = dummy_dataclass(frozen=False)
    obj.a.c = factor * obj.a.c
    obj.a.d = factor * obj.a.d
    obj.b = factor * obj.b
    target_obj = dummy_dataclass(factor=factor, frozen=False)
    asserts.assert_trees_all_close(obj, target_obj)

  def test_frozen_dataclass_raise_error(self):
    factor = 5.
    obj = dummy_dataclass(frozen=True)
    obj.a.c = factor * obj.a.c  # mutable since obj.a is not frozen.
    with self.assertRaisesRegex(dataclass.FrozenInstanceError,
                                'cannot assign to field'):
      obj.b = factor * obj.b  # raises error because obj is frozen.

  @parameterized.named_parameters(
      ('frozen', True),
      ('mutable', False),
  )
  def test_get_and_set_state(self, frozen):

    @chex_dataclass(frozen=frozen)
    class SimpleClass():
      data: int = 1

    obj_a = SimpleClass(data=1)
    state = obj_a.__getstate__()
    obj_b = SimpleClass(data=2)
    obj_b.__setstate__(state)
    self.assertEqual(obj_a, obj_b)

  def test_unexpected_kwargs(self):

    @chex_dataclass()
    class SimpleDataclass:
      a: int
      b: int = 2

    SimpleDataclass(a=1, b=3)
    with self.assertRaisesRegex(ValueError, 'init.*got unexpected kwargs'):
      SimpleDataclass(a=1, b=3, c=4)

  def test_tuple_conversion(self):

    @chex_dataclass()
    class SimpleDataclass:
      b: int
      a: int

    obj = SimpleDataclass(a=2, b=1)
    self.assertSequenceEqual(obj.to_tuple(), (1, 2))

    obj2 = SimpleDataclass.from_tuple((1, 2))
    self.assertEqual(obj.a, obj2.a)
    self.assertEqual(obj.b, obj2.b)

  @parameterized.named_parameters(
      ('frozen', True),
      ('mutable', False),
  )
  def test_tuple_rev_conversion(self, frozen):
    obj = dummy_dataclass(frozen=frozen)
    asserts.assert_trees_all_close(type(obj).from_tuple(obj.to_tuple()), obj)

  @parameterized.named_parameters(
      ('frozen', True),
      ('mutable', False),
  )
  def test_inheritance(self, frozen):

    @chex_dataclass(frozen=frozen)
    class Base:
      x: int

    @chex_dataclass(frozen=frozen)
    class Derived(Base):
      y: int

    base_obj = Base(x=1)
    self.assertNotIsInstance(base_obj, Derived)
    self.assertIsInstance(base_obj, Base)

    derived_obj = Derived(x=1, y=2)
    self.assertIsInstance(derived_obj, Derived)
    self.assertIsInstance(derived_obj, Base)

  def test_inheritance_from_empty_frozen_base(self):

    @chex_dataclass(frozen=True)
    class FrozenBase:
      pass

    @chex_dataclass(frozen=True)
    class DerivedFrozen(FrozenBase):
      j: int

    df = DerivedFrozen(j=2)
    self.assertIsInstance(df, FrozenBase)

    with self.assertRaisesRegex(
        TypeError, 'cannot inherit non-frozen dataclass from a frozen one'):

      # pylint:disable=unused-variable
      @chex_dataclass
      class DerivedMutable(FrozenBase):
        j: int

      # pylint:enable=unused-variable

  def test_disallowed_fields(self):
    # pylint:disable=unused-variable
    with self.assertRaisesRegex(ValueError, 'dataclass fields are disallowed'):

      @chex_dataclass(mappable_dataclass=False)
      class InvalidNonMappable:
        from_tuple: int

    @chex_dataclass(mappable_dataclass=False)
    class ValidMappable:
      get: int

    with self.assertRaisesRegex(ValueError, 'dataclass fields are disallowed'):

      @chex_dataclass(mappable_dataclass=True)
      class InvalidMappable:
        get: int
        from_tuple: int

    # pylint:enable=unused-variable

  @parameterized.parameters(True, False)
  def test_flatten_is_leaf(self, is_mappable):

    @chex_dataclass(mappable_dataclass=is_mappable)
    class _InnerDcls:
      v_1: int
      v_2: int

    @chex_dataclass(mappable_dataclass=is_mappable)
    class _Dcls:
      str_val: str
      # pytype: disable=invalid-annotation  # enable-bare-annotations
      inner_dcls: _InnerDcls
      dct: Mapping[str, _InnerDcls]
      # pytype: enable=invalid-annotation  # enable-bare-annotations

    dcls = _Dcls(
        str_val='test',
        inner_dcls=_InnerDcls(v_1=1, v_2=11),
        dct={
            'md1': _InnerDcls(v_1=2, v_2=22),
            'md2': _InnerDcls(v_1=3, v_2=33)
        })

    def _is_leaf(value) -> bool:
      # Must not traverse over integers.
      self.assertNotIsInstance(value, int)
      return isinstance(value, (_InnerDcls, str))

    leaves = jax.tree_util.tree_flatten(dcls, is_leaf=_is_leaf)[0]
    self.assertCountEqual(
        (dcls.str_val, dcls.inner_dcls, dcls.dct['md1'], dcls.dct['md2']),
        leaves)

    asserts.assert_tree_all_equal_structs(
        jax.tree_util.tree_map(lambda x: x, dcls, is_leaf=_is_leaf), dcls)

  @parameterized.named_parameters(
      ('mappable', True),
      ('not_mappable', False),
  )
  def test_generic_dataclass(self, mappable):
    # Running under Python 3.6 results in error "TypeError: Cannot inherit from
    # plain Generic", I'm speculating that this is a bug in cpython since
    # subsequent versions work
    if sys.version_info < (3, 7):
      self.skipTest('Skip test on Python version < 3.7')

    T = TypeVar('T')

    @chex_dataclass(mappable_dataclass=mappable)
    class GenericDataclass(Generic[T]):
      a: T  # pytype: disable=invalid-annotation  # enable-bare-annotations

    obj = GenericDataclass(a=np.array([1.0, 1.0]))
    asserts.assert_tree_all_close(obj.a, 1.0)

  def test_mappable_eq_override(self):

    @chex_dataclass(mappable_dataclass=True)
    class EqDataclass:
      a: pytypes.ArrayDevice

      def __eq__(self, other):
        if isinstance(other, EqDataclass):
          return other.a[0] == self.a[0]
        return False

    obj1 = EqDataclass(a=np.array([1.0, 1.0]))
    obj2 = EqDataclass(a=np.array([1.0, 0.0]))
    obj3 = EqDataclass(a=np.array([0.0, 1.0]))
    self.assertEqual(obj1, obj2)
    self.assertNotEqual(obj1, obj3)


if __name__ == '__main__':
  absltest.main()
