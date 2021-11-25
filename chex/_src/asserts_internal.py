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
import dataclasses
import functools
import re
import threading
import traceback
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Type, Union

from absl import logging
from chex._src import pytypes
import jax
from jax.experimental import host_callback as hcb
import jax.numpy as jnp
import numpy as np

_lock = threading.Lock()
_internal_error_message = (
    "Internal correctness check failed. Please file an issue on Chex bug "
    "tracker and describe how to reproduce this failure.")

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
TJittableAssertFn = Callable[..., bool]

# Matchers.
TDimMatcher = Optional[Union[int, Set[int], type(Ellipsis)]]
TShapeMatcher = Sequence[TDimMatcher]

# Chex namespace variables.
ERR_PREFIX: str = "[Chex] "
TRACE_COUNTER: collections.Counter = collections.Counter()
DISABLE_ASSERTIONS: bool = False

# Chex assertions internals.
IS_CHEX_CONTEXT: bool = False
ERRORS_REGISTRY: List["ValueAssertionError"] = []
VALUE_ASSERTION_METADATA_REGISTRY: List["ValueAssertionMetadata"] = []
VALUE_ASSERTION_DEPS_CHAIN: Optional["DependencyChain"] = None


@dataclasses.dataclass(frozen=True)
class ValueAssertionMetadata:
  """A dataclass to store a value assertion' metadata."""
  assertion_id: int
  stack_trace: Sequence[str]
  custom_message: Optional[str]
  include_default_message: bool
  exception_type: Type[Exception]

  def format_stack_trace(self) -> str:
    return "\n".join(self.stack_trace)


@dataclasses.dataclass(frozen=True)
class ValueAssertionError:
  """A dataclass to store the information about a failed value assertion."""
  assertion_id: int
  message: str
  device: pytypes.Device

  def format_error(self) -> str:
    metadata = VALUE_ASSERTION_METADATA_REGISTRY[self.assertion_id]
    stack_trace = metadata.format_stack_trace()
    return f"{stack_trace}{self.message} [on device '{self.device}']"


class DependencyChain:
  """A class that maintains a chain of (HLO) control dependencies.

  Used to inject additional dependencies into the outputs of the jitted
  functions to work around DCE optimization for value assertions.
  """

  def __init__(self) -> None:
    self._head_elem = None
    self._is_attached = False

  def append(self, tree: pytypes.ArrayTree) -> None:
    """Appends the tree leaves to the chain."""
    if self._is_attached:
      raise RuntimeError("Dependency chain has already been attached.")
    for x in jax.tree_leaves(tree):
      if self._head_elem is None:
        self._head_elem = x
      else:
        self._head_elem = hcb.id_tap_dep_p.bind(self._head_elem, x)

  def attach_to(self, tree: pytypes.ArrayTree) -> pytypes.ArrayTree:
    self._is_attached = True
    if self._head_elem is None:
      return tree
    else:
      return jax.tree_map(lambda x: hcb.id_tap_dep_p.bind(x, self._head_elem),
                          tree)


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


def _extract_stack_trace():
  return traceback.format_stack()[-6:-3]  # last 3 external frames


def _extract_errors_from_registry() -> Exception:
  with _lock:
    errors = [err.format_error() for err in ERRORS_REGISTRY]
    ERRORS_REGISTRY.clear()
    return AssertionError("The following errors were detected:\n" +
                          ("\n" + "-" * 40 + "\n").join(errors))


def make_value_assertion(
    host_assertion: TChexAssertion,
    jittable_assert_fn: TJittableAssertFn) -> TChexAssertion:
  """Constructs a value assertion.

  Value assertions can be used in compiled (i.e. jitted) functions.

  Args:
    host_assertion: an assertion to use on host.
    jittable_assert_fn: an assertion to use on device.

  Returns:
    A chex value assertion.
  """

  def _host_failure_handler(arg, transforms, *, device=None):
    del transforms

    (assertion_id, custom_message_format_vars, args, kwargs) = arg
    if not assertion_id < len(VALUE_ASSERTION_METADATA_REGISTRY):
      # All assertions must be registered after tracing.
      raise RuntimeError(_internal_error_message)
    assertion_meta = VALUE_ASSERTION_METADATA_REGISTRY[assertion_id]

    error_msg = None
    try:
      host_assertion(
          *args,
          custom_message=assertion_meta.custom_message,
          custom_message_format_vars=custom_message_format_vars,
          include_default_message=assertion_meta.include_default_message,
          exception_type=assertion_meta.exception_type,
          **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      error_msg = str(e)

    if error_msg is None:
      raise RuntimeError(f"Host could not reproduce the failure! Stacktrace:\n"
                         f"{assertion_meta.format_stacktrace()}")

    with _lock:
      ERRORS_REGISTRY.append(
          ValueAssertionError(
              assertion_id=assertion_id, message=error_msg, device=device))

  def _jittable_assertion(*args,
                          custom_message: Optional[str] = None,
                          custom_message_format_vars: Sequence[Any] = (),
                          include_default_message: bool = True,
                          exception_type: Type[Exception] = AssertionError,
                          **kwargs) -> None:
    # Save assertion's metadata during tracing.
    assertion_id = len(VALUE_ASSERTION_METADATA_REGISTRY)
    VALUE_ASSERTION_METADATA_REGISTRY.append(
        ValueAssertionMetadata(
            assertion_id=assertion_id,
            stack_trace=_extract_stack_trace(),
            custom_message=custom_message,
            include_default_message=include_default_message,
            exception_type=exception_type))

    # Tie new dependencies into tracers from the inputs.
    tie_in = get_tracers((args, kwargs))
    if not tie_in:
      # Value asserts should only be triggered for inputs with tracers.
      raise RuntimeError(_internal_error_message)

    def _passed_branch(tie_in):
      return tie_in

    def _failed_branch(tie_in):
      return hcb.id_tap(
          _host_failure_handler,
          arg=(assertion_id, custom_message_format_vars, args, kwargs),
          result=tie_in,
          tap_with_device=True)

    is_passed = jittable_assert_fn(*args, **kwargs)
    dependency = jax.lax.cond(is_passed, _passed_branch, _failed_branch, tie_in)
    VALUE_ASSERTION_DEPS_CHAIN.append(dependency)  # only during tracing

  return _jittable_assertion


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


def chex_assertion(
    assert_fn: TAssertFn,
    jittable_assert_fn: Optional[TJittableAssertFn] = None) -> TChexAssertion:
  """Wraps Chex assert functions to control their common behaviour.

  Extends the assertion to support the following optional auxiliary kwargs:
    custom_message: A string to include into the emitted exception messages.
    custom_message_format_vars: A list of variables to pass as arguments to
      `custom_message.format()`.
    include_default_message: Whether to include the default Chex message into
      the emitted exception messages.
    exception_type: An exception type to use. `AssertionError` by default.

  Args:
    assert_fn: a function implementing the check.
    jittable_assert_fn: an optional jittable version of `assert_fn`. Required
      for value assertions.

  Returns:
    A Chex assertion (with auxiliary kwargs).
  """

  host_assertion = make_static_assertion(assert_fn)
  is_value_assertion = (jittable_assert_fn or assert_fn) is not assert_fn
  if is_value_assertion:
    # We do not verify that `jittable_assert_fn` is jittable.
    value_assertion = make_value_assertion(host_assertion, jittable_assert_fn)

  @functools.wraps(assert_fn)
  def _chex_assertion(*args, **kwargs) -> None:
    if DISABLE_ASSERTIONS:
      return

    if is_value_assertion and has_tracers((args, kwargs)):
      if not IS_CHEX_CONTEXT:
        raise RuntimeError(
            "Value assertions can only be called from functions wrapped "
            "with `@chex.with_value_assertions`. See the docs.")

      assert VALUE_ASSERTION_DEPS_CHAIN is not None
      value_assertion(*args, **kwargs)
    else:
      host_assertion(*args, **kwargs)

  return _chex_assertion


def with_value_assertions(pure_fn, apply_transform, barrier_wait=True):
  """Wraps a jitted function to post-process the value assertions' results.

  Args:
    pure_fn: a pure (unjitted) python function to transform.
    apply_transform: a chain of JAX transformations to appy to `pure_fn`.
    barrier_wait: a bool suggesting whether to wait until all host callbacks
      complete before returning the outputs. Warning: prevents run-ahead exec.
      strategy, see https://jax.readthedocs.io/en/latest/async_dispatch.html.

  Returns:
    A transformed function that supports value assertions.
  """

  # Check wrappers ordering.
  if is_traceable(pure_fn):
    raise ValueError(
        "@with_value_assertions must not wrap JAX-transformed functions. "
        "Change wrappers ordering.")

  def _pure_fn_with_folded_deps(*args, **kwargs):
    if not IS_CHEX_CONTEXT:
      # This func should only be called from `_transformed_fn_with_sentinel`.
      raise RuntimeError(_internal_error_message)

    global VALUE_ASSERTION_DEPS_CHAIN
    if VALUE_ASSERTION_DEPS_CHAIN is not None:
      # Only this func is allowed to manipulate `VALUE_ASSERTION_DEPS_CHAIN`.
      raise RuntimeError(_internal_error_message)
    VALUE_ASSERTION_DEPS_CHAIN = DependencyChain()

    outputs = pure_fn(*args, **kwargs)
    outputs_with_deps = VALUE_ASSERTION_DEPS_CHAIN.attach_to(outputs)
    VALUE_ASSERTION_DEPS_CHAIN = None
    return outputs_with_deps

  transformed_fn = apply_transform(_pure_fn_with_folded_deps)

  @functools.wraps(pure_fn)
  def _transformed_fn_with_sentinel(*args, **kwargs):
    # Check on errors from the previous calls (e.g. when `barrier_wait==False`).
    if ERRORS_REGISTRY:
      raise _extract_errors_from_registry()

    global IS_CHEX_CONTEXT
    if IS_CHEX_CONTEXT:
      # Only this function is allowed to switch `IS_CHEX_CONTEXT`.
      raise RuntimeError(_internal_error_message)
    try:
      IS_CHEX_CONTEXT = True
      output = transformed_fn(*args, **kwargs)
    finally:
      IS_CHEX_CONTEXT = False
      global VALUE_ASSERTION_DEPS_CHAIN
      VALUE_ASSERTION_DEPS_CHAIN = None

    if barrier_wait:
      hcb.barrier_wait()  # Optional. Waits until all callbacks are completed.

    # Check on new errors.
    if ERRORS_REGISTRY:
      raise _extract_errors_from_registry()

    return output

  return _transformed_fn_with_sentinel


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
      x for x in jax.tree_leaves(tree) if isinstance(x, jax.core.Tracer))


def has_tracers(tree: pytypes.ArrayTree) -> bool:
  """Checks whether a tree contains any tracers."""
  return any(isinstance(x, jax.core.Tracer) for x in jax.tree_leaves(tree))


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
