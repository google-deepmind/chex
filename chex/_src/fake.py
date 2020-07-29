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
"""Utilities to patch JAX functions with faked implementations.

This module provides fake implementations of jax.jit and jax.pmap, which can be
patched over existing implementations for easier debugging.

See https://www.martinfowler.com/articles/mocksArentStubs.html
"""

import contextlib
import functools
import os
import re
from typing import Optional
from unittest import mock
from absl import flags
import jax
import jax.numpy as jnp


FLAGS = flags.FLAGS
flags.DEFINE_integer('chex_n_cpu_devices', 1,
                     'Number of CPU threads to use as devices in tests.')
flags.DEFINE_bool('chex_assert_multiple_cpu_devices', False,
                  'Whether to fail if a number of CPU devices is less than 2.')

_xla_device_count_flag_regexp = (
    r'[-]{0,2}xla_force_host_platform_device_count=(\d+)?(\s|$)')


def get_n_cpu_devices_from_xla_flags():
  """Parses number of CPUs from the XLA environment flags."""
  m = re.match(_xla_device_count_flag_regexp, os.getenv('XLA_FLAGS', ''))

  # At least one CPU device must be available.
  n_devices = int(m.group(1)) if m else 1
  return n_devices


def set_n_cpu_devices(n: Optional[int] = None):
  """Forces XLA to use `n` CPU threads as host devices.

  This allows `jax.pmap` to be tested on a single-CPU platform.
  This utility only takes effect before XLA backends are initialized, i.e.
  before any JAX operation is executed (including `jax.devices()` etc.).
  See https://github.com/google/jax/issues/1408.

  Args:
    n: required number of CPU devices (`FLAGS.chex_n_cpu_devices` if `None`).

  Raises:
    RuntimeError: if XLA backends were already initialized.
  """
  n = n or FLAGS.chex_n_cpu_devices

  n_devices = get_n_cpu_devices_from_xla_flags()
  cpu_backend = (jax.lib.xla_client._local_backends or {}).get('cpu', None)  # pylint: disable=protected-access
  if cpu_backend is not None and n_devices != n:
    raise RuntimeError(
        f'Attempted to set {n} devices, but {n_devices} CPUs already available:'
        ' ensure that `set_n_cpu_devices` is executed before any JAX operation.'
    )

  xla_flags = os.getenv('XLA_FLAGS', '')
  xla_flags = re.sub(_xla_device_count_flag_regexp, '', xla_flags)
  os.environ['XLA_FLAGS'] = ' '.join(
      [f'--xla_force_host_platform_device_count={n}'] + xla_flags.split())


@functools.wraps(jax.jit)
def _fake_jit(fn, *unused_args, **unused_kwargs):
  return fn


@functools.wraps(jax.pmap)
def _fake_pmap(fn, *unused_args, **unused_kwargs):
  return jax.vmap(fn)


def _zero(*unused_args, **unused_kwargs):
  return 0


def _identity(x, *unused_args, **unused_kwargs):
  return x


_fake_axis_index = functools.wraps(jax.lax.axis_index)(_zero)
_fake_psum = functools.wraps(jax.lax.psum)(_identity)
_fake_pmean = functools.wraps(jax.lax.pmean)(_identity)
_fake_pmax = functools.wraps(jax.lax.pmax)(_identity)
_fake_pmin = functools.wraps(jax.lax.pmin)(_identity)


@functools.wraps(jax.lax.all_gather)
def _fake_all_gather(x, *unused_args, **unused_kwargs):
  add_leading_dim = lambda t: t[jnp.newaxis]
  return jax.tree_map(add_leading_dim, x)


def fake_jit(enable_patching: bool = True):
  """Context manager for patching jax.jit with the identity function.

  This is intended to be used as a debugging tool to programmatically enable or
  disable JIT compilation.

  Can be used either as a context managed scope:

    with chex.fake_jit():
      @jax.jit
      def foo(x):
        ...

  or by calling `start` and `stop`:

    fake_jit_context = chex.fake_jit()
    fake_jit.context.start()
    @jax.jit
      def foo(x):
        ...
    fake_jit.context.stop()

  Args:
    enable_patching: Whether to patch jax.jit

  Returns:
    Context where jax.jit is patched with the identity function
  """
  stack = contextlib.ExitStack()
  if enable_patching:
    stack.enter_context(mock.patch('jax.jit', _fake_jit))
  return stack


def fake_pmap(enable_patching: bool = False):
  """Context manager for patching jax.pmap with jax.vmap.

  This is intended to be used as a debugging tool to programmatically replace
  pmap transformations with an equivalent non-parallel vmap transformation.

  Can be used either as a context managed scope:

    with chex.fake_pmap():
      @jax.pmap
      def foo(x):
        ...

  or by calling `start` and `stop`:

    fake_pmap_context = chex.fake_pmap()
    fake_pmap.context.start()
    @jax.pmap
      def foo(x):
        ...
    fake_pmap.context.stop()

  Args:
    enable_patching: Whether to patch jax.pmap

  Returns:
    Context where jax.pmap is patched with jax.vmap
  """
  # Improve implementation to automatically track JAX collectives development.
  stack = contextlib.ExitStack()
  if enable_patching:
    stack.enter_context(mock.patch('jax.pmap', _fake_pmap))
    stack.enter_context(mock.patch('jax.lax.axis_index', _fake_axis_index))
    stack.enter_context(mock.patch('jax.lax.psum', _fake_psum))
    stack.enter_context(mock.patch('jax.lax.pmean', _fake_pmean))
    stack.enter_context(mock.patch('jax.lax.pmax', _fake_pmax))
    stack.enter_context(mock.patch('jax.lax.pmin', _fake_pmin))
    stack.enter_context(mock.patch('jax.lax.all_gather', _fake_all_gather))
  return stack


def fake_pmap_and_jit(enable_pmap_patching: bool = True,
                      enable_jit_patching: bool = True):
  """Context manager for patching jax.jit and jax.pmap.

  This is a convenience function, equivalent to nested `chex.fake_pmap` and
  `chex.fake_jit` contexts.

  Args:
    enable_pmap_patching: Whether to patch jax.pmap
    enable_jit_patching: Whether to patch jax.jit

  Returns:
    Context where jax.pmap and jax.jit are patched with jax.vmap and the
    identity function
  """
  stack = contextlib.ExitStack()
  stack.enter_context(fake_pmap(enable_pmap_patching))
  stack.enter_context(fake_jit(enable_jit_patching))
  return stack
