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
"""A context manager that objects to JAX compilation for specified backends.

This is useful, for example, when certain JAX code needs to run in an
environment where an accelerator is present but reserved for other purposes.
Typically one would use `jax.jit(..., backend='cpu')` to keep the code away
from the accelerator, but it is hard to check by hand that this has been done
without exception throughout an entire subsystem. Then, `restrict_backends()`
can be used to detect any overlooked case and report it by raising an exception.

Similarly, it can be useful for a system such as a learner to make sure that
all required JAX programs have been assigned to their respective backends by
the end of its first iteration; this helps to show that it will not later run
into memory fragmentation problems. By entering a `restrict_backends()` context
at the end of the first iteration, the system can detect any overlooked cases.
"""
import contextlib
import functools
from typing import Callable, Optional, Sequence

from jax._src import compiler


class RestrictedBackendError(RuntimeError):
  pass


@contextlib.contextmanager
def _restrict_by_attr_name(name: str, is_allowed: Callable[[str], bool]):
  """Patch the given backend restriction onto getattr(compiler, name)."""
  inner_backend_compile = getattr(compiler, name)

  @functools.wraps(inner_backend_compile)
  def wrapper(backend, *args, **kwargs):
    if not is_allowed(backend.platform):
      raise RestrictedBackendError(
          f'Compiling a JAX program for {backend.platform} is forbidden by '
          f'restrict_backends().')
    return inner_backend_compile(backend, *args, **kwargs)

  try:
    setattr(compiler, name, wrapper)
    yield
  finally:
    backend_compile = getattr(compiler, name)
    assert backend_compile is wrapper, backend_compile
    setattr(compiler, name, inner_backend_compile)


@contextlib.contextmanager
def restrict_backends(
    *,
    allowed: Optional[Sequence[str]] = None,
    forbidden: Optional[Sequence[str]] = None):
  """Disallows JAX compilation for certain backends.

  Args:
    allowed: Names of backend platforms (e.g. 'cpu' or 'tpu') for which
      compilation is still to be permitted.
    forbidden: Names of backend platforms for which compilation is to be
      forbidden.

  Yields:
    None, in a context where compilation for forbidden platforms will raise
    a `RestrictedBackendError`.

  Raises:
    ValueError: if neither `allowed` nor `forbidden` is specified (i.e. they
      are both `None`), or if anything is both allowed and forbidden.
  """
  allowed = tuple(allowed) if allowed is not None else None
  forbidden = tuple(forbidden) if forbidden is not None else None

  if allowed is None and forbidden is None:
    raise ValueError('No restrictions specified.')
  contradictions = set(allowed or ()) & set(forbidden or ())
  if contradictions:
    raise ValueError(
        f"Backends {contradictions} can't be both allowed and forbidden.")

  def is_allowed(backend_platform):
    return (
        (backend_platform in allowed)
        if allowed is not None
        else (backend_platform not in forbidden)
    )

  with contextlib.ExitStack() as stack:
    # This is for compatibility with JAX both before and after
    # https://github.com/jax-ml/jax/commit/06448864abd6e8187e5b4d9b1ff08ab14fe3b8e0
    if hasattr(compiler, 'backend_compile_and_load'):
      stack.enter_context(
          _restrict_by_attr_name('backend_compile_and_load', is_allowed))
    elif hasattr(compiler, 'backend_compile'):
      stack.enter_context(_restrict_by_attr_name('backend_compile', is_allowed))
    yield
