# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Chex assertion internal utilities and symbols.

[README!]

We reserve the right to change the code in this module at any time without
providing any guarantees of backward compatibility. For this reason,
we strongly recommend that you avoid using this module directly at all costs!
Instead, consider opening an issue on GitHub and describing your use case.
"""

import collections
import collections.abc
import functools
import re
from typing import Any, Sequence, Union, Callable, Optional, Set, Tuple, Type

from absl import logging
from chex._src import pytypes
import jax
import jax.numpy as jnp
import numpy as np

# Custom pytypes.
TLeaf = Any
TLeavesEqCmpFn = Callable[[TLeaf, TLeaf], bool]
TLeavesEqCmpErrorFn = Callable[[TLeaf, TLeaf], str]

# TODO(iukemaev): define a typing protocol for TChexAssertion.
# Chex assertion signature:
# (*args,
#  custom_message: Optional[str] = None,
#  custom_message_format_vars: Sequence[Any] = (),
#  include_default_message: bool = True,
#  exception_type: Type[Exception] = AssertionError,
#  **kwargs)
TChexAssertion = Callable[..., None]
TAssertFn = Callable[..., None]

# Matchers.
TDimMatcher = Optional[Union[int, Set[int], type(Ellipsis)]]
TShapeMatcher = Sequence[TDimMatcher]

# Chex namespace variables.
ERR_PREFIX = "[Chex] "
TRACE_COUNTER = collections.Counter()
DISABLE_ASSERTIONS = False


def assert_collection_of_arrays(inputs: Sequence[pytypes.Array]):
  """Checks if ``inputs`` is a collection of arrays."""
  if not isinstance(inputs, collections.abc.Collection):
    raise ValueError(f"`inputs` is not a collection of arrays: {inputs}.")


def jnp_to_np_array(arr: pytypes.Array) -> pytypes.Array:
  """Converts `jnp.ndarray` to `np.ndarray`."""
  if isinstance(arr, jnp.ndarray):
    if arr.dtype == jnp.bfloat16:
      # Numpy does not support `bfloat16`.
      return np.asarray(arr, np.float32)
    else:
      return np.asarray(arr, arr.dtype)
  else:
    return arr


def deprecation_wrapper(new_fn, old_name, new_name):
  """Allows deprecated functions to continue running, with a warning logged."""

  def inner_fn(*args, **kwargs):
    logging.warning(
        "chex.%s has been renamed to chex.%s, please update your code.",
        old_name, new_name)
    return new_fn(*args, **kwargs)

  return inner_fn


def get_err_regex(message: str) -> str:
  """Constructs a regexp for the exception message.

  Args:
    message: an exception message.

  Returns:
    Regexp that ensures the message follows the standard chex formatting.
  """
  # (ERR_PREFIX + any symbols (incl. \n) + message)
  return f"{re.escape(ERR_PREFIX)}[\\s\\S]*{message}"


def make_static_assertion(assert_fn: TAssertFn) -> TChexAssertion:
  """Constructs a static assertion from an assert function.

  This wrapper should be used for assertions that do not check values or are
  not used in jitted code.

  Args:
    assert_fn: a function implementing the check.

  Returns:
    A chex assertion.
  """

  def _static_assert(*args,
                     custom_message: Optional[str] = None,
                     custom_message_format_vars: Sequence[Any] = (),
                     include_default_message: bool = True,
                     exception_type: Type[Exception] = AssertionError,
                     **kwargs) -> None:
    # Format error's stack trace to remove Chex' internal frames.
    assertion_exc = None
    value_exc = None
    try:
      assert_fn(*args, **kwargs)
    except AssertionError as e:
      assertion_exc = e
    except ValueError as e:
      value_exc = e
    finally:
      if value_exc is not None:
        raise ValueError(str(value_exc))

      if assertion_exc is not None:
        # Format the exception message.
        error_msg = str(assertion_exc)

        # Include only the name of the outermost chex assertion.
        if error_msg.startswith(ERR_PREFIX):
          error_msg = error_msg[error_msg.find("failed:") + len("failed:"):]

        # Whether to include the default error message.
        default_msg = (f"Assertion {assert_fn.__name__} failed: "
                       if include_default_message else "")
        error_msg = f"{ERR_PREFIX}{default_msg}{error_msg}"

        # Whether to include a custom error message.
        if custom_message:
          if custom_message_format_vars:
            custom_message = custom_message.format(*custom_message_format_vars)
          error_msg = f"{error_msg} [{custom_message}]"

        raise exception_type(error_msg)

  return _static_assert


def chex_assertion(assert_fn: TAssertFn,
                   value_assertion: bool) -> TChexAssertion:
  """Wraps Chex assert functions to control their common behaviour.

  Extends the assertion to support the following optional auxiliary kwargs:
    custom_message: A string to include into the emitted exception messages.
    custom_message_format_vars: A list of variables to pass as arguments to
      `custom_message.format()`.
    include_default_message: Whether to include the default Chex message into
      the emitted exception messages.
    exception_type: An exception type to use. `AssertionError` by default.

  Args:
    assert_fn: A function implementing the check.
    value_assertion: A bool indicating whether the assertion depends on actual
      tensors' values.

  Returns:
    A Chex assertion (with auxiliary kwargs).
  """
  host_assertion = make_static_assertion(assert_fn)

  @functools.wraps(assert_fn)
  def _chex_assert_fn(*args, **kwargs) -> None:
    if DISABLE_ASSERTIONS:
      return

    if value_assertion and has_tracers((args, kwargs)):
      cte_url = ("https://jax.readthedocs.io/en/latest/errors.html"
                 "#jax.errors.ConcretizationTypeError")
      raise RuntimeError(
          f"Assertions that use tensors' values cannot be called from a "
          f"jax-traced code. See {cte_url}.")
    else:
      try:
        host_assertion(*args, **kwargs)
      except jax.errors.ConcretizationTypeError as exc:
        msg = ("Chex assertion detected `ConcretizationTypeError`: it is very "
               "likely that it tried to access tensors' values during tracing. "
               "Make sure that you defined a jittable version of this Chex "
               "assertion.")
        raise exc from RuntimeError(msg)

  return _chex_assert_fn


def format_tree_path(path: Sequence[Any]) -> str:
  return "/".join(str(p) for p in path)


def format_shape_matcher(shape: TShapeMatcher) -> str:
  return f"({', '.join('...' if d is Ellipsis else str(d) for d in shape)})"


def num_devices_available(devtype: str, backend: Optional[str] = None) -> int:
  """Returns the number of available device of the given type."""
  devtype = devtype.lower()
  supported_types = ("cpu", "gpu", "tpu")
  if devtype not in supported_types:
    raise ValueError(
        f"Unknown device type '{devtype}' (expected one of {supported_types}).")

  return sum(d.platform == devtype for d in jax.devices(backend))


def get_tracers(tree: pytypes.ArrayTree) -> Tuple[jax.core.Tracer]:
  """Returns a tuple with tracers from a tree."""
  return tuple(
      x for x in jax.tree_util.tree_leaves(tree)
      if isinstance(x, jax.core.Tracer))


def has_tracers(tree: pytypes.ArrayTree) -> bool:
  """Checks whether a tree contains any tracers."""
  return any(
      isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def is_traceable(fn) -> bool:
  """Checks if function is traceable.

  JAX traces a function when it is wrapped with @jit, @pmap, or @vmap.
  In other words, this function checks whether `fn` is wrapped with any of
  the aforementioned JAX transformations.

  Args:
    fn: function to assert.

  Returns:
    Bool indicating whether fn is traceable.
  """

  fn_string_tokens = (
      "_python_jit.",  # PyJIT  in Python ver. < 3.7
      "_cpp_jit.",  # CppJIT in Python ver. < 3.7 (deprecated)
      ".reraise_with_filtered_traceback",  # JIT    in Python ver. >= 3.7
      "CompiledFunction",  # C++ JIT in jaxlib 0.1.66 or newer.
      "pmap.",  # Python pmap
      "PmapFunction",  # C++ pmap in jaxlib 0.1.72 or newer.
      "vmap.",  # vmap
  )

  fn_type_tokens = (
      "CompiledFunction",
      "PmapFunction",
  )

  # Un-wrap `fn` and check if any internal fn is jitted by pattern matching.
  fn_ = fn
  while True:
    if any(t in str(fn_) for t in fn_string_tokens):
      return True

    if any(t in str(type(fn_)) for t in fn_type_tokens):
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

      try:
        if isinstance(fn_, jax.lib.xla_extension.jax_jit.CompiledFunction):
          return True
      except AttributeError:
        pass
    else:
      break

    fn_ = fn_.__wrapped__
  return False


def assert_leaves_all_eq_comparator(
    equality_comparator: TLeavesEqCmpFn,
    error_msg_fn: Callable[[TLeaf, TLeaf, str, int, int],
                           str], path: Sequence[Any], *leaves: Sequence[TLeaf]):
  """Asserts all leaves are equal using custom comparator."""
  path_str = format_tree_path(path)
  for i in range(1, len(leaves)):
    if not equality_comparator(leaves[0], leaves[i]):
      raise AssertionError(error_msg_fn(leaves[0], leaves[i], path_str, 0, i))
