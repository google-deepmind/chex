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
"""Chex assertion utilities."""
import collections
import collections.abc
import functools
import inspect
import itertools
import traceback
from typing import Any, Sequence, Type, Union, Callable, Optional, Set
import unittest
from unittest import mock

from chex._src import pytypes
import jax
import jax.numpy as jnp
import jax.test_util as jax_test
import numpy as np
import tree as dm_tree

Scalar = pytypes.Scalar
Array = pytypes.Array
ArrayTree = pytypes.ArrayTree

# Custom pytypes.
TLeaf = Any
TLeavesEqCmpFn = Callable[[TLeaf, TLeaf], bool]
TLeavesEqCmpErrorFn = Callable[[TLeaf, TLeaf], str]


def _num_devices_available(devtype: str, backend: Optional[str] = None) -> int:
  """Returns the number of available device of the given type."""
  devtype = devtype.lower()
  supported_types = ("cpu", "gpu", "tpu")
  if devtype not in supported_types:
    raise ValueError(
        f"Unknown device type '{devtype}' (expected one of {supported_types}).")

  return sum(d.platform == devtype for d in jax.devices(backend))


def _is_traceable(fn):
  """Checks if function is traceable.

  JAX traces a function when it is wrapped with @jit, @pmap, or @vmap.
  In other words, this function checks whether `fn` is wrapped with any of
  the aforementioned JAX transformations.

  Args:
    fn: function to assert.

  Returns:
    Bool indicating whether fn is traceable.
  """

  tokens = (
      "_python_jit.",  # PyJIT  in Python ver. < 3.7
      "_cpp_jit.",  # CppJIT in Python ver. < 3.7
      ".reraise_with_filtered_traceback",  # JIT    in Python ver. >= 3.7
      "pmap.",  # pmap
      "vmap.",  # vmap
  )

  # Un-wrap `fn` and check if any internal f-n is jitted by pattern matching.
  fn_ = fn
  while True:
    if any(t in str(fn_) for t in tokens):
      return True

    if hasattr(fn_, "__wrapped__"):
      # Wrapper.
      fn_globals = getattr(fn_, "__globals__", {})

      if fn_globals.get("__name__", None) == "jax.api":
        # Wrapper from `jax.api`.
        return True

      if "api_boundary" in fn_globals:
        # api_boundary is a JAX wrapper for traced functions.
        return True
    else:
      break

    fn_ = fn_.__wrapped__
  return False


def _assert_leaves_all_eq_comparator(
    equality_comparator: TLeavesEqCmpFn,
    error_msg_fn: Callable[[TLeaf, TLeaf, str, int, int],
                           str], path: Sequence[Any], *leaves: Sequence[TLeaf]):
  """Asserts all leaves are equal using custom comparator."""
  path_str = "/".join(str(p) for p in path)
  for i in range(1, len(leaves)):
    if not equality_comparator(leaves[0], leaves[i]):
      raise AssertionError(error_msg_fn(leaves[0], leaves[i], path_str, 0, i))


_TRACE_COUNTER = collections.Counter()


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
          f"Function '{fn.__name__}' is traced > {n} times!\n"
          "It often happens when a jitted function is defined inside another "
          "function that is called multiple times (i.e. the jitted f-n is a "
          "new object every time). Make sure that your code does not exploit "
          "this pattern (move the nested functions to the top level to fix it)."
          " See `chex.clear_trace_counter()` if `@chex.assert_max_traces` is "
          "used in unittests."
      )

    return fn(*args, **kwargs)

  return fn_wrapped


def assert_scalar(x: Scalar):
  """Checks argument is a scalar, as defined in pytypes.py (int or float)."""
  if not isinstance(x, (int, float)):
    raise AssertionError(
        "The argument must be a scalar, was {}".format(type(x)))


def assert_scalar_in(
    x: Scalar, min_: Scalar, max_: Scalar, included: bool = True):
  assert_scalar(x)
  if included:
    if not min_ <= x <= max_:
      raise AssertionError(
          "The argument must be in [{}, {}], was {}".format(min_, max_, x))
  else:
    if not min_ < x < max_:
      raise AssertionError(
          "The argument must be in ({}, {}), was {}".format(min_, max_, x))


def assert_scalar_positive(x: Scalar):
  """Checks that the scalar is strictly positive."""
  assert_scalar(x)
  if x <= 0:
    raise AssertionError("The argument must be positive, was {}".format(x))


def assert_scalar_non_negative(x: Scalar):
  """Checks that the scalar is non negative."""
  assert_scalar(x)
  if x < 0:
    raise AssertionError("The argument must be non negative, was {}".format(x))


def assert_scalar_negative(x: Scalar):
  """Checks that the scalar is non negative."""
  assert_scalar(x)
  if x >= 0:
    raise AssertionError("The argument must be negative, was {}".format(x))


def assert_equal_shape(inputs: Sequence[Array]):
  """Checks that all arrays have the same shape.

  Args:
    inputs: sequence of arrays.

  Raises:
    AssertionError: if the shapes of all arrays do not match.
  """
  if isinstance(inputs, collections.abc.Sequence):
    shape = inputs[0].shape
    expected_shapes = [shape] * len(inputs)
    shapes = [x.shape for x in inputs]
    if shapes != expected_shapes:
      raise AssertionError(f"Arrays have different shapes: {shapes}.")


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


def assert_shape(inputs: Union[Scalar, Union[Array, Sequence[Array]]],
                 expected_shapes: Union[Sequence[Optional[int]],
                                        Sequence[Sequence[Optional[int]]]]):
  """Checks that the shape of all inputs matches specified expected_shapes.

  Valid usages include:

  ```
    assert_shape(x, ())                    # x is scalar
    assert_shape(x, (2, 3))                # x has shape (2, 3)
    assert_shape(x, (2, None))             # x has rank 2 and `x.shape[0] == 2`
    assert_shape([x, y], ())               # x and y are scalar
    assert_shape([x, y], [(), (2,3)])      # x is scalar and y has shape (2, 3)
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
  if isinstance(expected_shapes, Array):
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
    if not (
        len(shape) == len(expected)
        and all(j is None or i == j for i, j in zip(shape, expected))):
      errors.append((idx, shape, expected))

  if errors:
    msg = "; ".join(
        "input {} has shape {} but expected {}".format(*err) for err in errors)
    raise AssertionError("Error in shape compatibility check: " + msg + ".")


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
  if isinstance(expected_ranks, Array):
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
        "input {} has rank {} (shape {}) but expected {}".format(*err)
        for err in errors)

    raise AssertionError("Error in rank compatibility check: " + msg + ".")


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
        "input {} has type {} but expected {}".format(*err) for err in errors)

    raise AssertionError("Error in type compatibility check: " + msg + ".")


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
        "Expected tensor to have dimension {} along the axis {}"
        " but got {} instead.".format(expected, axis, tensor.shape[axis]))


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


def assert_tree_no_nones(tree: ArrayTree):
  """Asserts tree does not contain `None`s.

  Args:
    tree: a tree to assert.

  Raises:
    AssertionError: if the tree contains `None`s.
  """

  def _assert_fn(path, leaf):
    if leaf is None:
      formatted_path = "/".join(str(p) for p in path)
      raise AssertionError(f"`None` detected at '{formatted_path}'.")

  dm_tree.map_structure_with_path(_assert_fn, tree)


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
          f"Tree leaf '{'/'.join(path)}' has a shape prefix "
          f"diffent from expected: {prefix} != {shape_prefix}.")

  dm_tree.map_structure_with_path(_assert_fn, tree)


def assert_tree_all_equal_structs(*trees: Sequence[ArrayTree]):
  """Asserts trees have the same structure.

  Note that `None`s are treated as PyTree nodes.

  Args:
    *trees: trees which structures to assert on equality.

  Raise:
    AssertionError: if structures of any two trees are different.
  """
  first_treedef = jax.tree_structure(trees[0])
  other_treedefs = (jax.tree_structure(t) for t in trees[1:])
  for i, treedef in enumerate(other_treedefs, start=1):
    if first_treedef != treedef:
      raise AssertionError(
          f"Error in tree structs equality check: trees 0 and {i} do not match,"
          f"\n tree 0: {first_treedef}"
          f"\n tree {i}: {treedef}")


def assert_tree_all_equal_comparator(equality_comparator: TLeavesEqCmpFn,
                                     error_msg_fn: TLeavesEqCmpErrorFn,
                                     *trees: Sequence[ArrayTree],
                                     ignore_nones: bool = False):
  """Asserts all trees are equal as per the custom comparator for leaves.

  Args:
    equality_comparator: a custom function that accepts two leaves and checks
                         whether they are equal. Expected to be transitive.
    error_msg_fn: a function accepting two unequal as per `equality_comparator`
                  leaves and returning an error message.
    *trees: trees to check on equality as per `equality_comparator`.
    ignore_nones: whether to ignore `None`s in the trees.

  Raises:
    AssertionError: if `equality_comparator` returns False on any pair of trees.
  """
  if len(trees) < 2: return
  assert_tree_all_equal_structs(*trees)
  if not ignore_nones: assert_tree_no_nones(trees)

  def tree_error_msg_fn(l_1: TLeaf, l_2: TLeaf, path: str, i_1: int, i_2: int):
    msg = error_msg_fn(l_1, l_2)
    return f"Trees {i_1} and {i_2} differ in leaves '{path}': {msg}."

  def wrapped_equality_comparator(leaf_1, leaf_2):
    if leaf_1 is None or leaf_1 is None:
      # Either both or none of leaves can be `None`s.
      assert leaf_1 is None and leaf_2 is None, (
          "non-mutual cases must be caught by assert_tree_all_equal_structs")
      if ignore_nones:
        return True

    return equality_comparator(leaf_1, leaf_2)

  cmp = functools.partial(_assert_leaves_all_eq_comparator,
                          wrapped_equality_comparator, tree_error_msg_fn)
  dm_tree.map_structure_with_path(cmp, *trees)


def assert_tree_all_close(*trees: Sequence[ArrayTree],
                          rtol: float = 1e-07,
                          atol: float = .0,
                          ignore_nones: bool = False):
  """Asserts trees have leaves with approximately equal values.

  This compares the difference between values of actual and desired to
   atol + rtol * abs(desired).

  Args:
    *trees: a sequence of trees with array leaves.
    rtol: relative tolerance.
    atol: absolute tolerance.
    ignore_nones: whether to ignore `None`s in the trees.

  Raise:
    AssertionError: if the leaf values actual and desired are not equal up to
      specified tolerance, or trees contain `None`s (with `ignore_nones=False`).
  """
  assert_fn = functools.partial(
      np.testing.assert_allclose, rtol=rtol, atol=atol,
      err_msg="Error in value equality check: Values not approximately equal")

  def cmp_fn(arr_1, arr_2):
    assert_fn(arr_1, arr_2)  # Raises an AssertionError if values are not close.
    return True

  dummy_err_msg_fn = lambda arr_1, arr_2: None
  assert_tree_all_equal_comparator(
      cmp_fn, dummy_err_msg_fn, *trees, ignore_nones=ignore_nones)


def assert_tree_all_equal_shapes(*trees: Sequence[ArrayTree],
                                 ignore_nones: bool = False):
  """Asserts trees have the same structure and leaves' shapes.

  Args:
    *trees: a sequence of trees with array leaves.
    ignore_nones: whether to ignore `None`s in the trees.

  Raises:
    AssertionError: if trees' structures or leaves' shapes are different;
                    if trees contain `None`s (with `ignore_nones=False`).
  """
  cmp_fn = lambda arr_1, arr_2: arr_1.shape == arr_2.shape
  err_msg_fn = lambda arr_1, arr_2: f"shapes: {arr_1.shape} != {arr_2.shape}"
  assert_tree_all_equal_comparator(
      cmp_fn, err_msg_fn, *trees, ignore_nones=ignore_nones)


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


def assert_tpu_available(backend: Optional[str] = None):
  """Checks that at least one TPU device is available.

  Args:
    backend: a type of backend to use (use JAX default if `None`).

  Raises:
    AssertionError: if no TPU device available.
  """
  if not _num_devices_available("tpu", backend=backend):
    raise AssertionError(f"No TPU devices available in {jax.devices(backend)}.")


def assert_gpu_available(backend: Optional[str] = None):
  """Checks that at least one GPU device is available.

  Args:
    backend: a type of backend to use (use JAX default if `None`).

  Raises:
    AssertionError: if no GPU device available.
  """
  if not _num_devices_available("gpu", backend=backend):
    raise AssertionError(f"No GPU devices available in {jax.devices(backend)}.")


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


def assert_equal(first, second):
  """Assert the two objects are equal as determined by the '==' operator.

  Arrays with more than one element cannot be compared.
  Use `assert_tree_all_close` to compare arrays.

  Args:
    first: first object.
    second: second object.

  Raises:
    AssertionError: if not (first == second)
  """
  testcase = unittest.TestCase()
  testcase.assertEqual(first, second)


def assert_not_both_none(x, y):
  """Assert that at least one of the arguments is not `None`."""
  if x is None and y is None:
    raise ValueError(
        "At least one of the arguments must be different from `None`")


def assert_exactly_one_is_none(x, y):
  """Assert that one and only one of the arguments is `None`."""
  if (x is None) == (y is None):
    raise ValueError("Must pass one of the arguments, and not both.")


def if_args_not_none(fn, *args, **kwargs):
  """Wrap chex assertion to only be evaluated if positional args not None."""
  found_none = False
  for x in args:
    found_none = found_none or (x is None)
  if not found_none:
    fn(*args, **kwargs)
