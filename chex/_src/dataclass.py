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
"""JAX/dm-tree friendly dataclass implementation reusing Python dataclasses."""

import collections
import dataclasses
import functools
import inspect

from absl import logging
import jax
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet


FrozenInstanceError = dataclasses.FrozenInstanceError
_RESERVED_DCLS_FIELD_NAMES = frozenset(("from_tuple", "replace", "to_tuple"))


def _make_mappable(cls):
  """Create type that implements and inherits from ``collections.abc.Mapping``.

  Note that this does not require the class to be a dataclass, as it is supposed
  to be applied before creating the dataclass.

  Allows to traverse dataclasses in methods from `dm-tree` library.

  Args:
    cls: A class to use as a base for the new type.

  Returns:
    type implementing and inheriting from ``collections.abc.Mapping``.
  """
  # Define methods for compatibility with `collections.abc.Mapping`.
  setattr(cls, "__getitem__", lambda self, x: self.__dict__[x])
  setattr(cls, "__len__", lambda self: len(self.__dict__))
  setattr(cls, "__iter__", lambda self: iter(self.__dict__))

  # Update base class to derive from Mapping
  dct = dict(cls.__dict__)
  if "__dict__" in dct:
    dct.pop("__dict__")  # Avoid self-references.

  # Remove object from the sequence of base classes. Deriving from both Mapping
  # and object will cause a failure to create a MRO for the updated class
  bases = tuple(b for b in cls.__bases__ if b != object)
  return type(cls.__name__, bases + (collections.abc.Mapping,), dct)


def _convert_kw_only_dataclass_init(dcls):
  """Create wrapped initializer that converts everything to keyword arguments.

  This should be equivalent to passing `kw_only=True` when creating the
  dataclass in Python <= 3.10.

  Args:
    dcls: the dataclass to take the constructor from.

  Returns:
    Initializer wrapping the original initializer but which requires
    keyword-only arguments.

  Throws:
    ValueError: if all required arguments are not provided as keyword-only.
  """
  orig_init = dcls.__init__
  all_fields = set(f.name for f in dcls.__dataclass_fields__.values())
  init_fields = [f.name for f in dcls.__dataclass_fields__.values() if f.init]

  @functools.wraps(orig_init)
  def new_init(self, *orig_args, **orig_kwargs):
    if (orig_args and orig_kwargs) or len(orig_args) > 1:
      raise ValueError(
          "Mappable dataclass constructor doesn't support positional args."
          "(it has the same constructor as python dict)")
    all_kwargs = dict(*orig_args, **orig_kwargs)
    unknown_kwargs = set(all_kwargs.keys()) - all_fields
    if unknown_kwargs:
      raise ValueError(f"__init__() got unexpected kwargs: {unknown_kwargs}.")

    # Pass only arguments corresponding to fields with `init=True`.
    valid_kwargs = {k: v for k, v in all_kwargs.items() if k in init_fields}
    orig_init(self, **valid_kwargs)

  return new_init


def mappable_dataclass(cls):
  """Exposes dataclass as ``collections.abc.Mapping`` descendent.

  Allows to traverse dataclasses in methods from `dm-tree` library.

  NOTE: changes dataclasses constructor to dict-type
  (i.e. positional args aren't supported; however can use generators/iterables).

  Args:
    cls: A dataclass to mutate.

  Returns:
    Mutated dataclass implementing ``collections.abc.Mapping`` interface.
  """
  if not dataclasses.is_dataclass(cls):
    raise ValueError(f"Expected dataclass, got {cls} (change wrappers order?).")

  cls = _make_mappable(cls)
  cls.__init__ = _convert_kw_only_dataclass_init(cls)
  return cls


@dataclass_transform()
def dataclass(
    cls=None,
    *,
    init=True,
    repr=True,  # pylint: disable=redefined-builtin
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    mappable_dataclass=True,  # pylint: disable=redefined-outer-name
):
  """JAX-friendly wrapper for :py:func:`dataclasses.dataclass`.

  This wrapper class registers new dataclasses with JAX so that tree utils
  operate correctly. Additionally a replace method is provided making it easy
  to operate on the class when made immutable (frozen=True).

  Args:
    cls: A class to decorate.
    init: See :py:func:`dataclasses.dataclass`.
    repr: See :py:func:`dataclasses.dataclass`.
    eq: See :py:func:`dataclasses.dataclass`.
    order: See :py:func:`dataclasses.dataclass`.
    unsafe_hash: See :py:func:`dataclasses.dataclass`.
    frozen: See :py:func:`dataclasses.dataclass`.
    mappable_dataclass: If True (the default), methods to make the class
      implement the :py:class:`collections.abc.Mapping` interface will be
      generated and the class will include :py:class:`collections.abc.Mapping`
      in its base classes.
      `True` is the default, because being an instance of `Mapping` makes
      `chex.dataclass` compatible with e.g. `jax.tree_util.tree_*` methods, the
      `tree` library, or methods related to tensorflow/python/utils/nest.py.
      As a side-effect, e.g. `np.testing.assert_array_equal` will only check
      the field names are equal and not the content. Use `chex.assert_tree_*`
      instead.

  Returns:
    A JAX-friendly dataclass.
  """
  def dcls(cls):
    # Make sure to create a separate _Dataclass instance for each `cls`.
    return _Dataclass(
        init, repr, eq, order, unsafe_hash, frozen, mappable_dataclass
    )(cls)

  if cls is None:
    return dcls
  return dcls(cls)


class _Dataclass():
  """JAX-friendly wrapper for `dataclasses.dataclass`."""

  def __init__(
      self,
      init=True,
      repr=True,  # pylint: disable=redefined-builtin
      eq=True,
      order=False,
      unsafe_hash=False,
      frozen=False,
      mappable_dataclass=True,  # pylint: disable=redefined-outer-name
  ):
    self.init = init
    self.repr = repr  # pylint: disable=redefined-builtin
    self.eq = eq
    self.order = order
    self.unsafe_hash = unsafe_hash
    self.frozen = frozen
    self.mappable_dataclass = mappable_dataclass
    self.registered = False

  def __call__(self, cls):
    """Forwards class to dataclasses's wrapper and registers it with JAX."""

    if self.mappable_dataclass:
      cls = _make_mappable(cls)
      # We remove `collection.abc.Mapping` mixin methods here to allow
      # fields with these names.
      for attr in ("values", "keys", "get", "items"):
        setattr(cls, attr, None)  # redefine to avoid AttributeError on delattr
        delattr(cls, attr)        # delete

    # Remove once https://github.com/python/cpython/pull/24484 is merged.
    for base in cls.__bases__:
      if (dataclasses.is_dataclass(base) and
          getattr(base, "__dataclass_params__").frozen and not self.frozen):
        raise TypeError("cannot inherit non-frozen dataclass from a frozen one")

    # Check for invalid field names.
    annotations = inspect.get_annotations(cls)
    fields_names = set(name for name in annotations.keys())
    invalid_fields = fields_names.intersection(_RESERVED_DCLS_FIELD_NAMES)
    if invalid_fields:
      raise ValueError(f"The following dataclass fields are disallowed: "
                       f"{invalid_fields} ({cls}).")

    # pytype: disable=wrong-keyword-args
    dcls = dataclasses.dataclass(
        cls,
        init=self.init,
        repr=self.repr,
        eq=self.eq,
        order=self.order,
        # kw_only=self.mappable_dataclass,
        unsafe_hash=self.unsafe_hash,
        frozen=self.frozen)
    # pytype: enable=wrong-keyword-args

    def _from_tuple(args):
      return dcls(zip(dcls.__dataclass_fields__.keys(), args))

    def _to_tuple(self):
      return tuple(getattr(self, k) for k in self.__dataclass_fields__.keys())

    def _replace(self, **kwargs):
      return dataclasses.replace(self, **kwargs)

    def _getstate(self):
      return self.__dict__

    class_self = self

    # Patch __setstate__ to register the object on deserialization.
    def _setstate(self, state):
      if not class_self.registered:
        register_dataclass_type_with_jax_tree_util(dcls)
        class_self.registered = True
      self.__dict__.update(state)

    orig_init = dcls.__init__
    is_mappable_dataclass = self.mappable_dataclass
    if self.mappable_dataclass:
      kw_only_init = _convert_kw_only_dataclass_init(dcls)

    # Patch object's __init__ such that the class is registered on creation if
    # it is not registered on deserialization.
    @functools.wraps(orig_init)
    def _init(self, *args, **kwargs):
      if not class_self.registered:
        register_dataclass_type_with_jax_tree_util(dcls)
        class_self.registered = True

      if is_mappable_dataclass:
        return kw_only_init(self, *args, **kwargs)
      else:
        return orig_init(self, *args, **kwargs)

    setattr(dcls, "from_tuple", _from_tuple)
    setattr(dcls, "to_tuple", _to_tuple)
    setattr(dcls, "replace", _replace)
    setattr(dcls, "__getstate__", _getstate)
    setattr(dcls, "__setstate__", _setstate)
    setattr(dcls, "__init__", _init)

    return dcls


def _dataclass_unflatten(dcls, keys, values):
  """Creates a chex dataclass from a flatten jax.tree_util representation."""
  dcls_object = dcls.__new__(dcls)
  attribute_dict = dict(zip(keys, values))
  # Looping over fields instead of keys & values preserves the field order.
  # Using dataclasses.fields fails because dataclass uids change after
  # serialisation (eg, with cloudpickle).
  for field in dcls.__dataclass_fields__.values():
    if field.name in attribute_dict:  # Filter pseudo-fields.
      object.__setattr__(dcls_object, field.name, attribute_dict[field.name])
  # Need to manual call post_init here as we have avoided calling __init__
  if getattr(dcls_object, "__post_init__", None):
    dcls_object.__post_init__()
  return dcls_object


def _flatten_with_path(dcls):
  path = []
  keys = []
  for k, v in sorted(dcls.__dict__.items()):
    path.append((k, v))
    keys.append(k)
  return path, keys


def register_dataclass_type_with_jax_tree_util(data_class):
  """Register an existing dataclass so JAX knows how to handle it.

  This means that functions in jax.tree_util operate over the fields
  of the dataclass. See
  https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
  for further information.

  Args:
    data_class: A class created using dataclasses.dataclass. It must be
      constructable from keyword arguments corresponding to the members exposed
      in instance.__dict__.
  """
  flatten = lambda d: jax.util.unzip2(sorted(d.__dict__.items()))[::-1]
  unflatten = functools.partial(_dataclass_unflatten, data_class)
  try:
    jax.tree_util.register_pytree_with_keys(
        nodetype=data_class, flatten_with_keys=_flatten_with_path,
        flatten_func=flatten, unflatten_func=unflatten)
  except ValueError:
    logging.info("%s is already registered as JAX PyTree node.", data_class)
