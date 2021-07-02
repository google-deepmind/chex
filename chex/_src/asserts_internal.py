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

Do not import them into your project as they may (and will!) change over time.
"""

import collections
import collections.abc
import functools
import re
from typing import Any, Sequence, Union, Callable, Optional, Set

from absl import logging
import jax

# Custom pytypes.
TLeaf = Any
TLeavesEqCmpFn = Callable[[TLeaf, TLeaf], bool]
TLeavesEqCmpErrorFn = Callable[[TLeaf, TLeaf], str]

# Matchers.
TDimMatcher = Optional[Union[int, Set[int], type(Ellipsis)]]
TShapeMatcher = Sequence[TDimMatcher]

# Chex namespace variables.
ERR_PREFIX = "[Chex] "
TRACE_COUNTER = collections.Counter()


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


def chex_assertion(assert_fn) -> Callable[..., None]:
  """Wraps Chex assert functions to control their common behaviour."""

  @functools.wraps(assert_fn)
  def _wrapper(*args, **kwargs):
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
        error_msg = str(assertion_exc)
        # Include only the name of the outermost chex assertion.
        if error_msg.startswith(ERR_PREFIX):
          error_msg = error_msg[error_msg.find("failed:") + len("failed:"):]
        raise AssertionError(
            f"{ERR_PREFIX}Assertion {assert_fn.__name__} failed: {error_msg}")

  return _wrapper


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


def is_traceable(fn):
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
      "_cpp_jit.",  # CppJIT in Python ver. < 3.7 (deprecated)
      ".reraise_with_filtered_traceback",  # JIT    in Python ver. >= 3.7
      "CompiledFunction",  # C++ JIT in jaxlib 0.1.66 or newer.
      "pmap.",  # pmap
      "vmap.",  # vmap
  )

  # Un-wrap `fn` and check if any internal fn is jitted by pattern matching.
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
