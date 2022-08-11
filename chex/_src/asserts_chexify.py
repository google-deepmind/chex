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
"""Chexification utilities."""

import atexit
import collections
from concurrent import futures
import functools
from typing import Any, Callable

from absl import logging
from chex._src import asserts_internal as _ai
import jax
from jax.experimental import checkify


def _check_error(err: checkify.Error) -> None:
  try:
    checkify.check_error(err)
  except ValueError as exc:
    if str(exc).find(_ai.get_chexify_err_message()) != -1:
      raise AssertionError(str(exc))  # pylint:disable=raise-missing-from
    else:
      raise


def block_until_chexify_assertions_complete() -> None:
  """Waits until all asynchronous checks complete.

  See `chexify` for more detail.
  """
  for wait_fn in _ai.CHEXIFY_STORAGE.wait_fns:
    wait_fn()


@atexit.register  # to catch uninspected error stats
def _check_if_hanging_assertions():
  if _ai.CHEXIFY_STORAGE.wait_fns:
    logging.warning(
        '[Chex] Some of chexify assetion statuses were not inspected due to '
        'async exec (https://jax.readthedocs.io/en/latest/async_dispatch.html).'
        ' Consider calling `chex.block_until_chexify_assertions_complete()` at '
        'the end of computations that rely on jitted chex assetions.')
    block_until_chexify_assertions_complete()


def chexify(fn: Callable[..., Any],
            async_check: bool = True) -> Callable[..., Any]:
  """Wraps a transformed function `fn` to enable chex value assertions.

  Args:
    fn: A transformed function to wrap.
    async_check: Whether to check errors in the async dispatch mode. See
      https://jax.readthedocs.io/en/latest/async_dispatch.html.

  Returns:
    A _chexified_ function, i.e. the one with enabled value assertions.
    The returned function has `wait_checks()` method that blocks the caller
    until all pending async checks complete.
  """
  # Hardware/XLA failures can only happen on the C++ side. They are expected to
  # issue critical errors that will immediately crash the whole program.
  # Nevertheless, Chex sets its own timeout for every chexified XLA comp. to
  # ensure that a program never blocks on Chex side when running in async mode.
  async_timeout = 1800  # 30 minutes

  if async_check:
    # Spawn a thread for processing blocking calls.
    thread_pool = futures.ThreadPoolExecutor(1, f'async_chex_{fn.__name__}')
    # A deque for futures.
    async_check_futures = collections.deque()

  # Checkification.
  checkified_fn = checkify.checkify(fn)

  @functools.wraps(fn)
  def _chexified_fn(*args, **kwargs):
    if _ai.CHEXIFY_STORAGE.level:
      raise RuntimeError(
          'Nested @chexify wrapping is disallowed. '
          'Make sure that you only wrap the function at the outermost level.')

    if async_check:
      # Check completed calls.
      while async_check_futures and async_check_futures[0].done():
        _check_error(async_check_futures.popleft().result(async_timeout))

    # Run the checkified function.
    _ai.CHEXIFY_STORAGE.level += 1
    try:
      err, out = checkified_fn(*args, **kwargs)
    finally:
      _ai.CHEXIFY_STORAGE.level -= 1

    # Check errors.
    if async_check:
      # Blocking call is deferred to the thread.
      async_check_futures.append(
          thread_pool.submit(lambda: jax.device_get(err)))
    else:
      # Blocks until `fn`'s outputs are ready.
      _check_error(err)

    return out

  def _wait_checks():
    if async_check:
      while async_check_futures:
        _check_error(async_check_futures.popleft().result(async_timeout))

  # Add a barrier callback to the global storage.
  _ai.CHEXIFY_STORAGE.wait_fns.append(_wait_checks)

  # Add the callback to the chexified funtion's properties.
  if not hasattr(_chexified_fn, 'wait_checks'):
    _chexified_fn.wait_checks = _wait_checks
  else:
    logging.warning(
        "Function %s already defines 'wait_checks' method; "
        'Chex will not redefine it.', _chexified_fn.__name__)

  return _chexified_fn
