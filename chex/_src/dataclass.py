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
"""JAX friendly dataclass implementation reusing the dataclasses library."""

from absl import logging
import dataclasses
import jax
from jax import tree_util


FrozenInstanceError = dataclasses.FrozenInstanceError


def dataclass(  # pylint: disable=invalid-name
    _cls=None, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False,  # pylint: disable=redefined-builtin
    frozen=False):
  """"JAX-friendly wrapper for dataclasses.dataclass."""
  dcls = _Dataclass(
      init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash,
      frozen=frozen)
  if _cls is None:
    return dcls
  return dcls(_cls)


class _Dataclass():
  """"JAX-friendly wrapper for dataclasses.dataclass.

  This wrapper class registers new dataclasses with JAX so that tree utils
  operate correctly. Additionally a replace method is provided making it easy
  to operate on the class when made immutable (frozen=True).
  """

  def __init__(
      self, init=True, repr=True, eq=True, order=False, unsafe_hash=False,  # pylint: disable=redefined-builtin
      frozen=False):
    self.init = init
    self.repr = repr  # pylint: disable=redefined-builtin
    self.eq = eq
    self.order = order
    self.unsafe_hash = unsafe_hash
    self.frozen = frozen

  def __call__(self, cls):
    """Forwards class to dataclasses's wrapper and registers it with JAX."""
    dcls = dataclasses.dataclass(
        cls, init=self.init, repr=self.repr, eq=self.eq, order=self.order,
        unsafe_hash=self.unsafe_hash, frozen=self.frozen)  # pytype: disable=wrong-keyword-args

    def _replace(self, **kwargs):
      return dataclasses.replace(self, **kwargs)
    setattr(dcls, 'replace', _replace)

    _register_dataclass_type(dcls)
    return dcls


def _register_dataclass_type(data_class):
  """Register dataclass so JAX knows how to handle it."""
  flatten = lambda d: jax.tree_flatten(d.__dict__)
  unflatten = lambda s, xs: data_class(**s.unflatten(xs))
  try:
    tree_util.register_pytree_node(
        nodetype=data_class, flatten_func=flatten, unflatten_func=unflatten)
  except ValueError:
    logging.info('%s is already registered as JAX PyTree node.', data_class)
