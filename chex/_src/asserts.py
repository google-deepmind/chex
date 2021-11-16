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
"""Chex assertion utilities."""

import collections
import collections.abc
import functools
import inspect
import itertools
import traceback
from typing import Callable, List, Optional, Sequence, Set, Type, Union
import unittest
from unittest import mock

from chex._src import asserts_internal as ai
from chex._src import pytypes
import jax
import jax.numpy as jnp
import jax.test_util as jax_test
import numpy as np
import tree as dm_tree

Scalar = pytypes.Scalar
Array = pytypes.Array
ArrayTree = pytypes.ArrayTree

# Private symbols.
_ERR_PREFIX = ai.ERR_PREFIX
_TRACE_COUNTER = ai.TRACE_COUNTER

_TLeaf = ai.TLeaf
_TLeavesEqCmpFn = ai.TLeavesEqCmpFn
_TLeavesEqCmpErrorFn = ai.TLeavesEqCmpErrorFn
_ShapeMatcher = ai.TShapeMatcher

_assert_leaves_all_eq_comparator = ai.assert_leaves_all_eq_comparator
_chex_assertion = ai.chex_assertion
_format_shape_matcher = ai.format_shape_matcher
_format_tree_path = ai.format_tree_path
_is_traceable = ai.is_traceable
_num_devices_available = ai.num_devices_available


def disable_asserts():
  """Disables all Chex assertions.

  Use wisely.
  """
  ai.DISABLE_ASSERTIONS = True


def enable_asserts():
  """Enables Chex assertions."""
  ai.DISABLE_ASSERTIONS = False


def if_args_not_none(fn, *args, **kwargs):
  """Wrap chex assertion to only be evaluated if positional args not None."""
  found_none = False
  for x in args:
    found_none = found_none or (x is None)
  if not found_none:
    fn(*args, **kwargs)


def clear_trace_counter():
  """Clears Chex traces' counter for `assert_max_traces` checks.

  Use it to isolate unit tests that rely on `assert_max_traces`.
  """
  _TRACE_COUNTER.clear()


def assert_max_traces(fn=None, n=None):
  """Checks if a function is traced at most n times (inclusively).

  JAX re-traces JIT'ted function every time the structure of passed arguments
  changes. Often this behavior is inadvertent and leads to a significant
  performance drop which is hard to debug. This wrapper asserts that
  the function is not re-traced more that `n` times during program execution.

  Examples:

  ```
    @jax.jit
    @chex.assert_max_traces(n=1)
    def fn_sum_jitted(x, y):
      return x + y

    def fn_sub(x, y):
      return x - y

    fn_sub_pmapped = jax.pmap(chex.assert_max_retraces(fn_sub), n=10)
  ```

  More about tracing:
    https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html

  Args:
    fn: a function to wrap (must not be a JIT'ted function itself).
    n: maximum allowed number of retraces (non-negative).

  Returns:
    Decorated f-n that throws exception when max. number of re-traces exceeded.
  """
  if not callable(fn) and n is None:
    # Passed n as a first argument.
    n, fn = fn, n

  # Currying.
  if fn is None:
    return lambda fn_: assert_max_traces(fn_, n)

  assert_scalar_non_negative(n)

  # Check wrappers ordering.
  if _is_traceable(fn):
    raise ValueError(
        "@assert_max_traces must not wrap JAX-transformed function "
        "(@jit, @vmap, @pmap etc.); change wrappers ordering.")

  # Footprint is defined as a stacktrace of modules' names at the function's
  # definition place + its name and source code. This allows to catch retracing
  # event both in loops and in sequential calls. Can be used in Colab.
  fn_footprint = (
      tuple(frame.name for frame in traceback.extract_stack()[:-1]) +
      (inspect.getsource(fn), fn.__name__))
  fn_hash = hash(fn_footprint)

  @functools.wraps(fn)
  def fn_wrapped(*args, **kwargs):
    # We assume that a function without arguments is not being traced.
    # That is, case of n=0 for no-arguments function won't raise a error.
    has_tracers_in_args = any(
        isinstance(arg, jax.core.Tracer)
        for arg in itertools.chain(args, kwargs.values()))

    nonlocal fn_hash
    _TRACE_COUNTER[fn_hash] += int(has_tracers_in_args)
    if _TRACE_COUNTER[fn_hash] > n:
      raise AssertionError(
          f"{_ERR_PREFIX}Function '{fn.__name__}' is traced > {n} times!\n"
          "It often happens when a jitted function is defined inside another "
          "function that is called multiple times (i.e. the jitted f-n is a "
          "new object every time). Make sure that your code does not exploit "
          "this pattern (move the nested functions to the top level to fix it)."
          " See `chex.clear_trace_counter()` if `@chex.assert_max_traces` is "
          "used in unittests."
      )

    return fn(*args, **kwargs)

  return fn_wrapped


@_chex_assertion
def assert_scalar(x: Scalar):
  """Checks argument is a scalar, as defined in pytypes.py (int or float)."""
  if not isinstance(x, (int, float)):
    raise AssertionError(f"The argument must be a scalar, was {type(x)}.")


@_chex_assertion
def assert_scalar_in(
    x: Scalar, min_: Scalar, max_: Scalar, included: bool = True):
  """Checks argument is a scalar within segment (by default)."""
  assert_scalar(x)
  if included:
    if not min_ <= x <= max_:
      raise AssertionError(f"The argument must be in [{min_}, {max_}], was {x}")
  else:
    if not min_ < x < max_:
      raise AssertionError(f"The argument must be in ({min_}, {max_}), was {x}")


@_chex_assertion
def assert_scalar_positive(x: Scalar):
  """Checks that the scalar is strictly positive."""
  assert_scalar(x)
  if x <= 0:
    raise AssertionError(f"The argument must be positive, was {x}.")


@_chex_assertion
def assert_scalar_non_negative(x: Scalar):
  """Checks that the scalar is non negative."""
  assert_scalar(x)
  if x < 0:
    raise AssertionError(f"The argument must be non negative, was {x}.")


@_chex_assertion
def assert_scalar_negative(x: Scalar):
  """Checks that the scalar is non negative."""
  assert_scalar(x)
  if x >= 0:
    raise AssertionError(f"The argument must be negative, was {x}.")


@_chex_assertion
def assert_equal_shape(inputs: Sequence[Array], *,
                       dims: Optional[Union[int, Sequence[int]]] = None):
  """Checks that all arrays have the same shape.

  Args:
    inputs: sequence of arrays.
    dims: optional int or sequence of integers. If not provided, every dimension
      of every shape must match. If provided, equality of shape will only be
      asserted for the specified dim(s), i.e. to ensure all of a group of arrays
      have the same size in the first two dimensions, call
      `assert_equal_shape(tensors_list, dims=[0, 1])`.

  Raises:
    AssertionError: if the shapes of all arrays at specified dims do not match.
    ValueError: if the provided `dims` are invalid indices into any of `arrays`.
  """
  # NB: Need explicit dims argument, closing over it triggers linter bug.
  def extract_relevant_dims(shape, dims):
    try:
      if dims is None:
        return shape
      elif isinstance(dims, int):
        return shape[dims]
      else:
        return [shape[d] for d in dims]
    except IndexError as err:
      raise ValueError(
          f"Indexing error when trying to extra dim(s) {dims} from array shape "
          f"{shape}") from err

  if isinstance(inputs, collections.abc.Sequence):
    shape = extract_relevant_dims(inputs[0].shape, dims)
    expected_shapes = [shape] * len(inputs)
    shapes = [extract_relevant_dims(x.shape, dims) for x in inputs]
    if shapes != expected_shapes:
      if dims is not None:
        msg = f"Arrays have different shapes at dims {dims}: {shapes}"
      else:
        msg = f"Arrays have different shapes: {shapes}."
      raise AssertionError(msg)


@_chex_assertion
def assert_equal_shape_prefix(inputs, prefix_len):
  """Check that the leading `prefix_dims` dims of all inputs have same shape.

  Args:
    inputs: sequence of input arrays.
    prefix_len: number of leading dimensions to compare; each input's shape
      will be sliced to `shape[:prefix_len]`. Negative values are accepted
      and have the conventional Python indexing semantics.

  Raises:
    AssertionError: if the shapes of all arrays do not match.
  """
  shapes = [array.shape[:prefix_len] for array in inputs]
  if shapes != [shapes[0]] * len(shapes):
    raise AssertionError(f"Arrays have different shape prefixes: {shapes}")


@_chex_assertion
def assert_equal_shape_suffix(inputs, suffix_len):
  """Check that the final `suffix_len` dims of all inputs have same shape.

  Args:
    inputs: sequence of input arrays.
    suffix_len: number of trailing dimensions to compare; each input's shape
      will be sliced to `shape[-suffix_len:]`. Negative values are accepted
      and have the conventional Python indexing semantics.

  Raises:
    AssertionError: if the shapes of all arrays do not match.
  """
  shapes = [array.shape[-suffix_len:] for array in inputs]
  if shapes != [shapes[0]] * len(shapes):
    raise AssertionError(f"Arrays have different shape suffixes: {shapes}")


def _unelided_shape_matches(
    actual_shape: Sequence[int],
    expected_shape: Sequence[Optional[Union[int, Set[int]]]]) -> bool:
  """Returns True if `actual_shape` is compatible with `expected_shape`."""
  if len(actual_shape) != len(expected_shape):
    return False
  for actual, expected in zip(actual_shape, expected_shape):
    if expected is None:
      continue
    if isinstance(expected, set):
      if actual not in expected:
        return False
    elif actual != expected:
      return False
  return True


def _shape_matches(actual_shape: Sequence[int],
                   expected_shape: _ShapeMatcher) -> bool:
  """Returns True if `actual_shape` is compatible with `expected_shape`."""
  # Splits `expected_shape` based on the position of the ellipsis, if present.
  expected_prefix: List[Union[int, Set[int]]] = []
  expected_suffix: Optional[List[Union[int, Set[int]]]] = None
  for dim in expected_shape:
    if dim is Ellipsis:
      if expected_suffix is not None:
        raise ValueError(
            "`expected_shape` may not contain more than one ellipsis, "
            f"but got {_format_shape_matcher(expected_shape)}")
      expected_suffix = []
    elif expected_suffix is None:
      expected_prefix.append(dim)
    else:
      expected_suffix.append(dim)

  # If there is no ellipsis, just compare to the full `actual_shape`.
  if expected_suffix is None:
    assert len(expected_prefix) == len(expected_shape)
    return _unelided_shape_matches(actual_shape, expected_prefix)

  # Checks that the actual rank is least the number of non-elided dimensions.
  if len(actual_shape) < len(expected_prefix) + len(expected_suffix):
    return False

  if expected_prefix:
    actual_prefix = actual_shape[:len(expected_prefix)]
    if not _unelided_shape_matches(actual_prefix, expected_prefix):
      return False

  if expected_suffix:
    actual_suffix = actual_shape[-len(expected_suffix):]
    if not _unelided_shape_matches(actual_suffix, expected_suffix):
      return False

  return True


@_chex_assertion
def assert_shape(inputs: Union[Scalar, Union[Array, Sequence[Array]]],
                 expected_shapes: Union[_ShapeMatcher,
                                        Sequence[_ShapeMatcher]]):
  """Checks that the shape of all inputs matches specified expected_shapes.

  Valid usages include:

  ```
    assert_shape(x, ())                  # x is scalar
    assert_shape(x, (2, 3))              # x has shape (2, 3)
    assert_shape(x, (2, {1, 3}))         # x has shape (2, 1) or (2, 3)
    assert_shape(x, (2, None))           # x has rank 2 and `x.shape[0] == 2`
    assert_shape(x, (2, ...))            # x has rank >= 1 and `x.shape[0] == 2`
    assert_shape([x, y], ())             # x and y are scalar
    assert_shape([x, y], [(), (2,3)])    # x is scalar and y has shape (2, 3)
  ```

  Args:
    inputs: array or sequence of arrays.
    expected_shapes: sequence of expected shapes associated with each input,
      where the expected shape is a sequence of integer and `None` dimensions;
      if all inputs have same shape, a single shape may be passed as
      `expected_shapes`.

  Raises:
    AssertionError: if the length of `inputs` and `expected_shapes` don't match;
                    if `expected_shapes` has wrong type;
                    if shape of `input` does not match `expected_shapes`.
  """
  if isinstance(expected_shapes, (np.ndarray, Array)):
    raise AssertionError(
        "Error in shape compatibility check:"
        "Expected shapes should be a list or tuple of ints.")

  # Ensure inputs and expected shapes are sequences.
  if not isinstance(inputs, collections.abc.Sequence):
    inputs = [inputs]

  # Shapes are always lists or tuples, not scalars.
  if (not expected_shapes or not isinstance(expected_shapes[0], (list, tuple))):
    expected_shapes = [expected_shapes] * len(inputs)
  if len(inputs) != len(expected_shapes):
    raise AssertionError(
        "Length of `inputs` and `expected_shapes` must match: "
        f"{len(inputs)} is not equal to {len(expected_shapes)}.")

  errors = []
  for idx, (x, expected) in enumerate(zip(inputs, expected_shapes)):
    shape = getattr(x, "shape", ())  # scalars have shape () by definition.
    if not _shape_matches(shape, expected):
      errors.append((idx, shape, _format_shape_matcher(expected)))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has shape {e[1]} but expected {e[2]}" for e in errors)
    raise AssertionError(f"Error in shape compatibility check: {msg}.")


@_chex_assertion
def assert_is_broadcastable(shape_a: Sequence[int], shape_b: Sequence[int]):
  """Checks that an array of `shape_a` is broadcastable to one of `shape_b`.

  Args:
    shape_a: the shape of the array we want to broadcast.
    shape_b: the target shape after broadcasting.

  Raises:
    AssertionError: if `shape_a` is not broadcastable to `shape_b`.
  """
  error = AssertionError(
      f"Shape {shape_a} is not broadcastable to shape {shape_b}.")
  ndim_a = len(shape_a)
  ndim_b = len(shape_b)
  if ndim_a > ndim_b:
    raise error
  else:
    for i in range(1, ndim_a + 1):
      if shape_a[-i] != 1 and shape_a[-i] != shape_b[-i]:
        raise error


@_chex_assertion
def assert_equal_rank(inputs: Sequence[Array]):
  """Checks that all arrays have the same rank.

  Args:
    inputs: sequence of arrays.

  Raises:
    AssertionError: if the ranks of all arrays do not match.
  """
  if isinstance(inputs, collections.abc.Sequence):
    rank = len(inputs[0].shape)
    expected_ranks = [rank] * len(inputs)
    ranks = [len(x.shape) for x in inputs]
    if ranks != expected_ranks:
      raise AssertionError(f"Arrays have different rank: {ranks}.")


@_chex_assertion
def assert_rank(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_ranks: Union[int, Set[int], Sequence[Union[int, Set[int]]]]):
  """Checks that the rank of all inputs matches specified expected_ranks.

  Valid usages include:

  ```
    assert_rank(x, 0)                      # x is scalar
    assert_rank(x, 2)                      # x is a rank-2 array
    assert_rank(x, {0, 2})                 # x is scalar or rank-2 array
    assert_rank([x, y], 2)                 # x and y are rank-2 arrays
    assert_rank([x, y], [0, 2])            # x is scalar and y is a rank-2 array
    assert_rank([x, y], {0, 2})            # x and y are scalar or rank-2 arrays
  ```

  Args:
    inputs: array or sequence of arrays.
    expected_ranks: sequence of expected ranks associated with each input, where
      the expected rank is either an integer or set of integer options; if all
      inputs have same rank, a single scalar or set of scalars may be passed as
      `expected_ranks`.

  Raises:
    AssertionError: if the length of `inputs` and `expected_ranks` don't match;
                    if `expected_ranks` has wrong type;
                    if the ranks of input do not match the expected ranks.
  """
  if isinstance(expected_ranks, (np.ndarray, Array)):
    raise ValueError("Error in rank compatibility check: expected ranks should "
                     "be a collection of integers but was an array.")

  # Ensure inputs and expected ranks are sequences.
  if not isinstance(inputs, collections.abc.Sequence):
    inputs = [inputs]
  if (not isinstance(expected_ranks, collections.abc.Sequence) or
      isinstance(expected_ranks, collections.abc.Set)):
    expected_ranks = [expected_ranks] * len(inputs)
  if len(inputs) != len(expected_ranks):
    raise AssertionError(
        "Length of inputs and expected_ranks must match: inputs has length "
        f"{len(inputs)}, expected_ranks has length {len(expected_ranks)}.")

  errors = []
  for idx, (x, expected) in enumerate(zip(inputs, expected_ranks)):
    if hasattr(x, "shape"):
      shape = x.shape
    else:
      shape = ()  # scalars have shape () by definition.
    rank = len(shape)

    # Multiple expected options can be specified.

    # Check against old usage where options could be any sequence
    if (isinstance(expected, collections.abc.Sequence) and
        not isinstance(expected, collections.abc.Set)):
      raise ValueError("Error in rank compatibility check: "
                       "Expected ranks should be integers or sets of integers.")

    options = (
        expected if isinstance(expected, collections.abc.Set) else {expected})

    if rank not in options:
      errors.append((idx, rank, shape, expected))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has rank {e[1]} (shape {e[2]}) but expected {e[3]}"
        for e in errors)

    raise AssertionError(f"Error in rank compatibility check: {msg}.")


@_chex_assertion
def assert_type(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_types: Union[Type[Scalar], Sequence[Type[Scalar]]]):
  """Checks that the type of all `inputs` matches specified `expected_types`.

  Valid usages include:

  ```
    assert_type(7, int)
    assert_type(7.1, float)
    assert_type(False, bool)
    assert_type([7, 8], int)
    assert_type([7, 7.1], [int, float])
    assert_type(np.array(7), int)
    assert_type(np.array(7.1), float)
    assert_type(jnp.array(7), int)
    assert_type([jnp.array([7, 8]), np.array(7.1)], [int, float])
  ```

  Args:
    inputs: array or sequence of arrays or scalars.
    expected_types: sequence of expected types associated with each input; if
      all inputs have same type, a single type may be passed as
      `expected_types`.

  Raises:
    AssertionError: if the length of `inputs` and `expected_types` don't match;
                    if `expected_types` contains unsupported pytype;
                    if the types of input do not match the expected types.
  """
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  if not isinstance(expected_types, (list, tuple)):
    expected_types = [expected_types] * len(inputs)

  errors = []
  if len(inputs) != len(expected_types):
    raise AssertionError(f"Length of `inputs` and `expected_types` must match, "
                         f"got {len(inputs)} != {len(expected_types)}.")
  for idx, (x, expected) in enumerate(zip(inputs, expected_types)):
    if jnp.issubdtype(expected, jnp.floating):
      parent = jnp.floating
    elif jnp.issubdtype(expected, jnp.integer):
      parent = jnp.integer
    elif jnp.issubdtype(expected, jnp.bool_):
      parent = jnp.bool_
    else:
      raise AssertionError(
          f"Error in type compatibility check, unsupported dtype '{expected}'.")

    if not jnp.issubdtype(jnp.result_type(x), parent):
      errors.append((idx, jnp.result_type(x), expected))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has type {e[1]} but expected {e[2]}" for e in errors)

    raise AssertionError(f"Error in type compatibility check: {msg}.")


@_chex_assertion
def assert_axis_dimension(tensor: Array, axis: int, expected: int):
  """Assert dimension of a specific axis of a tensor.

  Args:
    tensor: a JAX array
    axis: an integer specifying which axis to assert.
    expected: expected value of `tensor.shape[axis]`.

  Raises:
    AssertionError: if the dimension of the specified axis does not match the
      prescribed value.
  """
  tensor = jnp.asarray(tensor)
  if axis >= len(tensor.shape) or axis < -len(tensor.shape):
    raise AssertionError(
        f"Expected tensor to have dim '{expected}' along axis '{axis}' but"
        f" axis '{axis}' not available: tensor rank is '{len(tensor.shape)}'.")
  if tensor.shape[axis] != expected:
    raise AssertionError(
        f"Expected tensor to have dimension {expected} along the axis {axis} "
        f"but got {tensor.shape[axis]} instead.")


@_chex_assertion
def assert_axis_dimension_gt(tensor: Array, axis: int, val: int):
  """Assert dimension of a specific axis of a tensor.

  Args:
    tensor: a JAX array.
    axis: an integer specifying which axis to assert.
    val: value `tensor.shape[axis]` must be greater than.

  Raises:
    AssertionError: if the dimension of `axis` is not greater than `val`.
  """
  tensor = jnp.asarray(tensor)
  if axis >= len(tensor.shape) or axis < -len(tensor.shape):
    raise AssertionError(
        f"Expected tensor to have dim greater than '{val}' on axis '{axis}' but"
        f" axis '{axis}' not available: tensor rank is '{len(tensor.shape)}'.")
  if tensor.shape[axis] <= val:
    raise AssertionError(
        f"Expected tensor to have dimension greater than '{val}' on axis"
        f" '{axis}' but got '{tensor.shape[axis]}' instead.")


@_chex_assertion
def assert_numerical_grads(
    f: Callable[..., Array],
    f_args: Sequence[Array],
    order: int,
    atol: float = 0.01,
    **check_kwargs):
  """Checks that autodiff and numerical gradients of a function match.

  Args:
    f: function to check.
    f_args: arguments of the function.
    order: order of gradients.
    atol: absolute tolerance.
    **check_kwargs: kwargs for `jax_test.check_grads`.

  Raises:
    AssertionError: if automatic differentiation gradients deviate from finite
    difference gradients.
  """
  # Correct scaling.
  # Remove after https://github.com/google/jax/issues/3130 is fixed.
  atol *= f_args[0].size

  # Mock `jax.lax.stop_gradient` because finite diff. method does not honour it.
  mock_sg = lambda t: jax.tree_map(jnp.ones_like, t)
  with mock.patch("jax.lax.stop_gradient", mock_sg):
    jax_test.check_grads(f, f_args, order=order, atol=atol, **check_kwargs)


@_chex_assertion
def assert_tree_no_nones(tree: ArrayTree):
  """Asserts tree does not contain `None`s.

  Args:
    tree: a tree to assert.

  Raises:
    AssertionError: if the tree contains `None`s.
  """

  def _assert_fn(path, leaf):
    if leaf is None:
      raise AssertionError(f"`None` detected at '{_format_tree_path(path)}'.")

  dm_tree.map_structure_with_path(_assert_fn, tree)


@_chex_assertion
def assert_tree_shape_prefix(tree: ArrayTree,
                             shape_prefix: Sequence[int],
                             *,
                             ignore_nones: bool = False):
  """Asserts all tree leaves' shapes have the same prefix.

  Args:
    tree: a tree to assert.
    shape_prefix: an expected shapes' prefix.
    ignore_nones: whether to ignore `None`s in the tree.
  Raise:
    AssertionError: if some leaf's shape doesn't start with the expected prefix;
                    if `ignore_nones` isn't set and the tree contains `None`s.
  """

  if not ignore_nones:
    assert_tree_no_nones(tree)

  def _assert_fn(path, leaf):
    if leaf is None: return

    prefix = leaf.shape[:len(shape_prefix)]
    if prefix != shape_prefix:
      raise AssertionError(
          f"Tree leaf '{_format_tree_path(path)}' has a shape prefix "
          f"different from expected: {prefix} != {shape_prefix}.")

  dm_tree.map_structure_with_path(_assert_fn, tree)


@_chex_assertion
def assert_tree_shape_suffix(tree: ArrayTree,
                             shape_suffix: Sequence[int],
                             *,
                             ignore_nones: bool = False):
  """Asserts all tree leaves' shapes have the same suffix.

  Args:
    tree: a tree to assert.
    shape_suffix: an expected shapes' suffix.
    ignore_nones: whether to ignore `None`s in the tree.
  Raise:
    AssertionError: if some leaf's shape doesn't start with the expected suffix;
                    if `ignore_nones` isn't set and the tree contains `None`s.
  """

  if not ignore_nones:
    assert_tree_no_nones(tree)

  def _assert_fn(path, leaf):
    if leaf is None: return
    if not shape_suffix: return  # No suffix, this is trivially true.

    if len(shape_suffix) > len(leaf.shape):
      raise AssertionError(
          f"Tree leaf '{_format_tree_path(path)}' has a shape suffix "
          f"of length {len(leaf.shape)} which is smaller than the expected "
          f"suffix of length {len(shape_suffix)}.")

    suffix = leaf.shape[-len(shape_suffix):]
    if suffix != shape_suffix:
      raise AssertionError(
          f"Tree leaf '{_format_tree_path(path)}' has a shape suffix "
          f"different from expected: {suffix} != {shape_suffix}.")

  dm_tree.map_structure_with_path(_assert_fn, tree)


@_chex_assertion
def assert_trees_all_equal_structs(*trees: ArrayTree):
  """Asserts trees have the same structure.

  Note that `None`s are treated as PyTree nodes.

  Args:
    *trees: >= 2 trees to assert equal structure between.

  Raise:
    AssertionError: if structures of any two trees are different.
  """
  if len(trees) < 2:
    raise ValueError(
        "assert_trees_all_equal_structs on a single tree does not make sense. "
        "Maybe you wrote `assert_trees_all_equal_structs([a, b])` instead of "
        "`assert_trees_all_equal_structs(a, b)` ?")
  first_treedef = jax.tree_structure(trees[0])
  other_treedefs = (jax.tree_structure(t) for t in trees[1:])
  for i, treedef in enumerate(other_treedefs, start=1):
    if first_treedef != treedef:
      raise AssertionError(
          f"Error in tree structs equality check: trees 0 and {i} do not match,"
          f"\n tree 0: {first_treedef}"
          f"\n tree {i}: {treedef}")


assert_tree_all_equal_structs = ai.deprecation_wrapper(
    assert_trees_all_equal_structs,
    old_name="assert_tree_all_equal_structs",
    new_name="assert_trees_all_equal_structs")


@_chex_assertion
def assert_trees_all_equal_comparator(equality_comparator: _TLeavesEqCmpFn,
                                      error_msg_fn: _TLeavesEqCmpErrorFn,
                                      *trees: ArrayTree,
                                      ignore_nones: bool = False):
  """Asserts all trees are equal as per the custom comparator for leaves.

  Args:
    equality_comparator: a custom function that accepts two leaves and checks
                         whether they are equal. Expected to be transitive.
    error_msg_fn: a function accepting two unequal as per `equality_comparator`
                  leaves and returning an error message.
    *trees: at least 2 trees to check on equality as per `equality_comparator`.
    ignore_nones: whether to ignore `None`s in the trees.

  Raises:
    ValueError: if *trees does not have at least two elements.
    AssertionError: if `equality_comparator` returns False on any pair of trees.
  """
  if len(trees) < 2:
    raise ValueError(
        "Assertions over only one tree does not make sense. Maybe you wrote "
        "`assert_trees_xxx([a, b])` instead of `assert_trees_xxx(a, b)`, or "
        "forgot the `error_msg_fn` arg to `assert_trees_all_equal_comparator`?")
  assert_trees_all_equal_structs(*trees)
  if not ignore_nones: assert_tree_no_nones(trees)

  def tree_error_msg_fn(l_1: _TLeaf, l_2: _TLeaf, path: str, i_1: int,
                        i_2: int):
    msg = error_msg_fn(l_1, l_2)
    if path:
      return f"Trees {i_1} and {i_2} differ in leaves '{path}': {msg}."
    else:
      return f"Trees (arrays) {i_1} and {i_2} differ: {msg}."

  def wrapped_equality_comparator(leaf_1, leaf_2):
    if leaf_1 is None or leaf_1 is None:
      # Either both or none of leaves can be `None`s.
      assert leaf_1 is None and leaf_2 is None, (
          "non-mutual cases must be caught by assert_trees_all_equal_structs")
      if ignore_nones:
        return True

    return equality_comparator(leaf_1, leaf_2)

  cmp = functools.partial(_assert_leaves_all_eq_comparator,
                          wrapped_equality_comparator, tree_error_msg_fn)
  dm_tree.map_structure_with_path(cmp, *trees)


assert_tree_all_equal_comparator = ai.deprecation_wrapper(
    assert_trees_all_equal_comparator,
    old_name="assert_tree_all_equal_comparator",
    new_name="assert_trees_all_equal_comparator")


@_chex_assertion
def assert_trees_all_equal_dtypes(*trees: ArrayTree,
                                  ignore_nones: bool = False):
  """Asserts trees' leaves have the same dtypes.

  Note that `None`s are treated as PyTree nodes.

  Args:
    *trees: >= 2 trees to assert equal types between.
    ignore_nones: whether to ignore `None`s in the trees.
  Raise:
    AssertionError: if leaves' types for any two trees are different.
  """

  def cmp_fn(arr_1, arr_2):
    return (hasattr(arr_1, "dtype") and hasattr(arr_2, "dtype") and
            arr_1.dtype == arr_2.dtype)

  def err_msg_fn(arr_1, arr_2):
    if not hasattr(arr_1, "dtype"):
      return f"{type(arr_1)} is not a (j-)np array (has no `dtype` property)"
    if not hasattr(arr_2, "dtype"):
      return f"{type(arr_2)} is not a (j-)np array (has no `dtype` property)"
    return f"types: {arr_1.dtype} != {arr_2.dtype}"

  assert_trees_all_equal_comparator(
      cmp_fn, err_msg_fn, *trees, ignore_nones=ignore_nones)


@_chex_assertion
def assert_trees_all_close(*trees: ArrayTree,
                           rtol: float = 1e-06,
                           atol: float = .0,
                           ignore_nones: bool = False):
  """Asserts trees have leaves with approximately equal values.

  This compares the difference between values of actual and desired to
   atol + rtol * abs(desired).

  Args:
    *trees: a sequence of >= 2 trees with array leaves.
    rtol: relative tolerance.
    atol: absolute tolerance.
    ignore_nones: whether to ignore `None`s in the trees.

  Raise:
    AssertionError: if the leaf values actual and desired are not equal up to
      specified tolerance, or trees contain `None`s (with `ignore_nones=False`).
  """

  def assert_fn(arr_1, arr_2):
    np.testing.assert_allclose(
        ai.jnp_to_np_array(arr_1),
        ai.jnp_to_np_array(arr_2),
        rtol=rtol,
        atol=atol,
        err_msg="Error in value equality check: Values not approximately equal")

  def cmp_fn(arr_1, arr_2) -> bool:
    try:
      # Raises an AssertionError if values are not close.
      assert_fn(arr_1, arr_2)
    except AssertionError:
      return False
    return True

  def err_msg_fn(arr_1, arr_2) -> str:
    try:
      assert_fn(arr_1, arr_2)
    except AssertionError as e:
      return (f"{str(e)} \nOriginal dtypes: "
              f"{np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}")
    return ""

  assert_trees_all_equal_comparator(
      cmp_fn, err_msg_fn, *trees, ignore_nones=ignore_nones)

assert_tree_all_close = ai.deprecation_wrapper(
    assert_trees_all_close,
    old_name="assert_tree_all_close",
    new_name="assert_trees_all_close")


@_chex_assertion
def assert_trees_all_equal(*trees: ArrayTree,
                           ignore_nones: bool = False):
  """Asserts trees have leaves with *exactly* equal values.

  If you are comparing floating point numbers, an exact equality check may not
  be appropriate; consider using assert_trees_all_close.

  Args:
    *trees: a sequence of >= 2 trees with array leaves.
    ignore_nones: whether to ignore `None`s in the trees.

  Raise:
    AssertionError: if the leaf values actual and desired are not exactly equal,
    or trees contain `None`s (with `ignore_nones=False`).
  """

  def assert_fn(arr_1, arr_2):
    np.testing.assert_array_equal(
        ai.jnp_to_np_array(arr_1),
        ai.jnp_to_np_array(arr_2),
        err_msg="Error in value equality check: Values not exactly equal")

  def cmp_fn(arr_1, arr_2) -> bool:
    try:
      # Raises an AssertionError if values are not equal.
      assert_fn(arr_1, arr_2)
    except AssertionError:
      return False
    return True

  def err_msg_fn(arr_1, arr_2) -> str:
    try:
      assert_fn(arr_1, arr_2)
    except AssertionError as e:
      return (f"{str(e)} \nOriginal dtypes: "
              f"{np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}")
    return ""

  assert_trees_all_equal_comparator(
      cmp_fn, err_msg_fn, *trees, ignore_nones=ignore_nones)


@_chex_assertion
def assert_trees_all_equal_shapes(*trees: ArrayTree,
                                  ignore_nones: bool = False):
  """Asserts trees have the same structure and leaves' shapes.

  Args:
    *trees: a sequence of >= 2 trees with array leaves.
    ignore_nones: whether to ignore `None`s in the trees.

  Raises:
    AssertionError: if trees' structures or leaves' shapes are different;
                    if trees contain `None`s (with `ignore_nones=False`).
  """
  cmp_fn = lambda arr_1, arr_2: arr_1.shape == arr_2.shape
  err_msg_fn = lambda arr_1, arr_2: f"shapes: {arr_1.shape} != {arr_2.shape}"
  assert_trees_all_equal_comparator(
      cmp_fn, err_msg_fn, *trees, ignore_nones=ignore_nones)


assert_tree_all_equal_shapes = ai.deprecation_wrapper(
    assert_trees_all_equal_shapes,
    old_name="assert_tree_all_equal_shapes",
    new_name="assert_trees_all_equal_shapes")


@_chex_assertion
def assert_devices_available(
    n: int,
    devtype: str,
    backend: Optional[str] = None,
    not_less_than: bool = False):
  """Checks that `n` devices of a given type are available.

  Args:
    n: required number of devices of a given type.
    devtype: type of devices, one of {'cpu', 'gpu', 'tpu'}.
    backend: type of backend to use (uses JAX default if `None`).
    not_less_than: whether to check if number of devices _not less_ than
      required `n` instead of precise comparison.

  Raises:
    AssertionError: if number of available device of a given type is not equal
    or less than `n`.
  """
  n_available = _num_devices_available(devtype, backend=backend)
  devs = jax.devices(backend)
  if not_less_than and n_available < n:
    raise AssertionError(
        f"Only {n_available} < {n} {devtype.upper()}s available in {devs}.")
  elif not not_less_than and n_available != n:
    raise AssertionError(f"No {n} {devtype.upper()}s available in {devs}.")


@_chex_assertion
def assert_tpu_available(backend: Optional[str] = None):
  """Checks that at least one TPU device is available.

  Args:
    backend: a type of backend to use (use JAX default if `None`).

  Raises:
    AssertionError: if no TPU device available.
  """
  if not _num_devices_available("tpu", backend=backend):
    raise AssertionError(f"No TPU devices available in {jax.devices(backend)}.")


@_chex_assertion
def assert_gpu_available(backend: Optional[str] = None):
  """Checks that at least one GPU device is available.

  Args:
    backend: a type of backend to use (use JAX default if `None`).

  Raises:
    AssertionError: if no GPU device available.
  """
  if not _num_devices_available("gpu", backend=backend):
    raise AssertionError(f"No GPU devices available in {jax.devices(backend)}.")


@_chex_assertion
def assert_tree_all_finite(tree_like: ArrayTree):
  """Assert all tensor leaves in a tree are finite.

  Args:
    tree_like: pytree with array leaves

  Raises:
    AssertionError: if any leaf in the tree is non-finite.
  """
  all_finite = jax.tree_util.tree_all(
      jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), tree_like))
  if not all_finite:
    is_finite = lambda x: "Finite" if jnp.all(jnp.isfinite(x)) else "Nonfinite"
    error_msg = jax.tree_map(is_finite, tree_like)
    raise AssertionError(f"Tree contains non-finite value: {error_msg}.")


@_chex_assertion
def assert_equal(first, second):
  """Assert the two objects are equal as determined by the '==' operator.

  Arrays with more than one element cannot be compared.
  Use `assert_trees_all_close` to compare arrays.

  Args:
    first: first object.
    second: second object.

  Raises:
    AssertionError: if not (first == second)
  """
  testcase = unittest.TestCase()
  testcase.assertEqual(first, second)


@_chex_assertion
def assert_not_both_none(x, y):
  """Assert that at least one of the arguments is not `None`."""
  if x is None and y is None:
    raise ValueError(
        "At least one of the arguments must be different from `None`")


@_chex_assertion
def assert_exactly_one_is_none(x, y):
  """Assert that one and only one of the arguments is `None`."""
  if (x is None) == (y is None):
    raise ValueError("Must pass one of the arguments, and not both.")


@_chex_assertion
def assert_is_divisible(numerator: int, denominator: int):
  """Assert that the numerator is divisible exactly by the denominator."""
  if numerator % denominator != 0:
    raise AssertionError(f"{numerator} is not divisible by {denominator}")
