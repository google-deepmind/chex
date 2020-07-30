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
import functools
from typing import Sequence, Type, Union, Callable, Optional, Set
from unittest import mock
from chex._src import pytypes
import jax
import jax.numpy as jnp
import jax.test_util as jax_test
import numpy as np

Scalar = pytypes.Scalar
Array = pytypes.Array
ArrayTree = pytypes.ArrayTree


def _num_devices_available(devtype: str, backend: Optional[str] = None):
  """Returns number of available device of the given type."""
  devtype = devtype.lower()
  supported_types = ("cpu", "gpu", "tpu")
  if devtype not in supported_types:
    raise ValueError(
        f"Unknown device type '{devtype}' (expected one of {supported_types}).")

  return sum(d.platform == devtype for d in jax.devices(backend))


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
  if isinstance(inputs, Sequence):
    shape = inputs[0].shape
    expected_shapes = [shape] * len(inputs)
    shapes = [x.shape for x in inputs]
    if shapes != expected_shapes:
      raise AssertionError(f"Arrays have different shapes: {shapes}.")


def assert_shape(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_shapes: Union[Sequence[int], Sequence[Sequence[int]]]):
  """Checks that the shape of all inputs matches specified expected_shapes.

  Valid usages include:

  ```
    assert_shape(x, ())                    # x is scalar
    assert_shape(x, (2, 3))                # x has shape (2, 3)
    assert_shape([x, y], ())               # x and y are scalar
    assert_shape([x, y], [(), (2,3)])      # x is scalar and y has shape (2, 3)
  ```

  Args:
    inputs: array or sequence of arrays.
    expected_shapes: sequence of expected shapes associated with each input,
      where the expected shape is a sequence of integer dimensions; if all
      inputs have same shape, a single shape may be passed as `expected_shapes`.

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
  if not isinstance(inputs, Sequence):
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
    if list(shape) != list(expected):
      errors.append((idx, shape, expected))

  if errors:
    msg = "; ".join(
        "input {} has shape {} but expected {}".format(*err) for err in errors)
    raise AssertionError("Error in shape compatibility check: " + msg + ".")


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
  if not isinstance(inputs, Sequence):
    inputs = [inputs]
  if (not isinstance(expected_ranks, Sequence) or
      isinstance(expected_ranks, Set)):
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
    if isinstance(expected, Sequence) and not isinstance(expected, Set):
      raise ValueError(
          "Error in rank compatibility check: "
          "Expected ranks should be integers or sets of integers.")

    options = expected if isinstance(expected, Set) else {expected}

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
    raise AssertionError(
        "Length of `inputs` and `expected_types` must match: `inputs` has length "
        f"{len(inputs)}, `expected_types` has length {len(expected_types)}.")
  for idx, (x, expected) in enumerate(zip(inputs, expected_types)):
    if jnp.issubdtype(expected, jnp.floating):
      parent = jnp.floating
    elif jnp.issubdtype(expected, jnp.integer):
      parent = jnp.integer
    else:
      raise AssertionError(
          f"Error in type compatibility check, unsupported dtype '{expected}'.")

    if not jnp.issubdtype(jnp.result_type(x), parent):
      errors.append((idx, jnp.result_type(x), expected))

  if errors:
    msg = "; ".join(
        "input {} has type {} but expected {}".format(*err) for err in errors)

    raise AssertionError("Error in type compatibility check: " + msg + ".")


def assert_numerical_grads(
    f: Callable[[Sequence[Array]], Array],
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


def assert_tree_all_close(
    actual: ArrayTree,
    desired: ArrayTree,
    rtol: float = 1e-07,
    atol: float = 0):
  """Assert two trees have leaves with approximately equal values.

  This compares the difference between values of actual and desired to
   atol + rtol * abs(desired).

  Args:
    actual: pytree with array leaves.
    desired: pytree with array leaves.
    rtol: relative tolerance.
    atol: absolute tolerance.
  Raise:
    AssertionError: if the leaf values actual and desired are not equal up to
      specified tolerance.
  """
  if jax.tree_structure(actual) != jax.tree_structure(desired):
    raise AssertionError(
        "Error in value equality check: Trees do not have the same structure,\n"
        f"actual: {jax.tree_structure(actual)}\n"
        f"desired: {jax.tree_structure(desired)}.")

  assert_fn = functools.partial(
      np.testing.assert_allclose,
      rtol=rtol,
      atol=atol,
      err_msg="Error in value equality check: Values not approximately equal")
  jax.tree_multimap(assert_fn, actual, desired)


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
