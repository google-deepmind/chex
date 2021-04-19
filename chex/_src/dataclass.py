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
import functools

from absl import logging
import dataclasses
import jax

FrozenInstanceError = dataclasses.FrozenInstanceError


def mappable_dataclass(cls, restricted_inheritance=True):
  """Exposes dataclass as `collections.abc.Mapping` descendent.

  Allows to traverse dataclasses in methods from `dm-tree` library.

  NOTE: changes dataclasses constructor to dict-type
  (i.e. positional args aren't supported; however can use generators/iterables).

  Args:
    cls: dataclass to mutate.
    restricted_inheritance: ensure dataclass inherits from `object` or from
      another `chex.dataclass`.

  Returns:
    Mutated dataclass implementing `collections.abc.Mapping` interface.
  """
  if not dataclasses.is_dataclass(cls):
    raise ValueError(f"Expected dataclass, got {cls} (change wrappers order?)")

  is_dataclass_base = all(map(dataclasses.is_dataclass, cls.__bases__))
  if (restricted_inheritance and (cls.__bases__ !=
                                  (object,) and not is_dataclass_base)):
    raise ValueError(
        f"Not a pure dataclass: undefined behaviour (bases: {cls.__bases__})."
        "Disable `restricted_inheritance` to suppress this check.")

  # Define methods for compatibility with `collections.abc.Mapping`.
  setattr(cls, "__getitem__", lambda self, x: self.__dict__[x])
  setattr(cls, "__len__", lambda self: len(self.__dict__))
  setattr(cls, "__iter__", lambda self: iter(self.__dict__))

  # Update constructor.
  orig_init = cls.__init__
  all_fields = set(f.name for f in cls.__dataclass_fields__.values())
  init_fields = [f.name for f in cls.__dataclass_fields__.values() if f.init]

  @functools.wraps(cls.__init__)
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

  cls.__init__ = new_init

  # Update base class.
  dct = dict(cls.__dict__)
  if "__dict__" in dct:
    dct.pop("__dict__")  # Avoid self-references.
  bases = tuple(b for b in cls.__bases__ if b != object)
  cls = type(cls.__name__, bases + (collections.abc.Mapping,), dct)
  return cls


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
    restricted_inheritance=True,
):
  """JAX-friendly wrapper for dataclasses.dataclass."""
  dcls = _Dataclass(init, repr, eq, order, unsafe_hash, frozen,
                    mappable_dataclass, restricted_inheritance)
  if cls is None:
    return dcls
  return dcls(cls)


class _Dataclass():
  """JAX-friendly wrapper for dataclasses.dataclass.

  This wrapper class registers new dataclasses with JAX so that tree utils
  operate correctly. Additionally a replace method is provided making it easy
  to operate on the class when made immutable (frozen=True).
  """

  def __init__(
      self,
      init=True,
      repr=True,  # pylint: disable=redefined-builtin
      eq=True,
      order=False,
      unsafe_hash=False,
      frozen=False,
      mappable_dataclass=True,  # pylint: disable=redefined-outer-name
      restricted_inheritance=True,
  ):
    self.init = init
    self.repr = repr  # pylint: disable=redefined-builtin
    self.eq = eq
    self.order = order
    self.unsafe_hash = unsafe_hash
    self.frozen = frozen
    self.mappable_dataclass = mappable_dataclass
    self.restricted_inheritance = restricted_inheritance

  def __call__(self, cls):
    """Forwards class to dataclasses's wrapper and registers it with JAX."""
    # pytype: disable=wrong-keyword-args
    dcls = dataclasses.dataclass(
        cls,
        init=self.init,
        repr=self.repr,
        eq=self.eq,
        order=self.order,
        unsafe_hash=self.unsafe_hash,
        frozen=self.frozen)
    # pytype: enable=wrong-keyword-args

    if self.mappable_dataclass:
      dcls = mappable_dataclass(dcls, self.restricted_inheritance)

    def _from_tuple(args):
      return dcls(zip(dcls.__dataclass_fields__.keys(), args))

    def _to_tuple(self):
      return tuple(getattr(self, k) for k in self.__dataclass_fields__.keys())

    def _replace(self, **kwargs):
      return dataclasses.replace(self, **kwargs)

    def _getstate(self):
      return self.__dict__

    def _setstate(self, state):
      self.__dict__.update(state)

    setattr(dcls, "from_tuple", _from_tuple)
    setattr(dcls, "to_tuple", _to_tuple)
    setattr(dcls, "replace", _replace)
    setattr(dcls, "__getstate__", _getstate)
    setattr(dcls, "__setstate__", _setstate)

    _register_dataclass_type(dcls)
    return dcls


def _register_dataclass_type(data_class):
  """Register dataclass so JAX knows how to handle it."""
  flatten = lambda d: jax.tree_flatten(d.__dict__)
  unflatten = lambda s, xs: data_class(**s.unflatten(xs))
  try:
    jax.tree_util.register_pytree_node(
        nodetype=data_class, flatten_func=flatten, unflatten_func=unflatten)
  except ValueError:
    logging.info("%s is already registered as JAX PyTree node.", data_class)
