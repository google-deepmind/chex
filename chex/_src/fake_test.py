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
"""Tests for `fake.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import fake
from chex._src import pytypes
import jax
import jax.numpy as jnp

ArrayBatched = pytypes.ArrayBatched
ArraySharded = pytypes.ArraySharded


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule():
  fake.set_n_cpu_devices()


def _assert_jitted(fn, fn_input, is_jitted):
  """Asserts that a function can be jitted or not.

  Args:
    fn: The function to be tested
    fn_input: Input to pass to the function
    is_jitted: Assert that the function can be jitted with jax.jit (True) or
    cannot be jitted (False), i.e. the fake jit is working correctly.
  """
  asserts.clear_trace_counter()
  max_traces = 1 if is_jitted else 0
  wrapped_fn = jax.jit(asserts.assert_max_traces(fn, max_traces))
  wrapped_fn(fn_input)


def _assert_pmapped(fn, fn_input, is_pmapped, should_jit=False):
  """Asserts whether a function can be pmapped or not.

  Args:
    fn: The function to be tested
    fn_input: Input to pass to the function
    is_pmapped: Assert that the function can be pmapped with jax.pmap (True) or
    cannot be pmapped (False), i.e. the fake pmap is working correctly.
    should_jit: if True, asserts that the function is jitted, regardless of it
    being pmapped or not.
  """
  num_devices = len(jax.devices())
  if should_jit:
    asserts.clear_trace_counter()
    fn = asserts.assert_max_traces(fn, n=1)
  wrapped_fn = jax.pmap(fn, axis_size=num_devices)

  fn_input = jnp.broadcast_to(fn_input, (num_devices,) + fn_input.shape)
  output = wrapped_fn(fn_input)

  # We test whether the function has been pmapped by inspecting the type of
  # the function output, if it is a sharded array type then the function has
  # been pmapped
  if not is_pmapped and hasattr(jax.interpreters.xla, 'type_is_device_array'):
    expected_type = 'DeviceArray'
    assert_message = f'Output is type {type(output)}, expected {expected_type}'
    assert jax.interpreters.xla.type_is_device_array(output), assert_message
  else:
    expected_type = ArraySharded if is_pmapped else jnp.DeviceArray
    assert_message = f'Output is type {type(output)}, expected {expected_type}'
    # We want to check exact types here
    assert type(output) == expected_type, assert_message    # pylint: disable=unidiomatic-typecheck


class PmapFakeTest(parameterized.TestCase):

  def test_assert_pmapped(self):
    def foo(x):
      return x * 2
    fn_input = jnp.ones((4,))

    _assert_pmapped(foo, fn_input, True)
    with self.assertRaises(AssertionError):
      _assert_pmapped(foo, fn_input, False)

  def test_assert_jitted(self):
    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    _assert_jitted(foo, fn_input, True)
    with self.assertRaises(AssertionError):
      _assert_jitted(foo, fn_input, False)

  @parameterized.named_parameters([
      ('plain_jit', {'enable_patching': True}, False),
      ('faked_jit', {'enable_patching': False}, True),
  ])
  def test_fake_jit(self, fake_kwargs, is_jitted):
    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    # Call with context manager
    with fake.fake_jit(**fake_kwargs):
      _assert_jitted(foo, fn_input, is_jitted)

    # Call with start/stop
    ctx = fake.fake_jit(**fake_kwargs)
    ctx.start()
    _assert_jitted(foo, fn_input, is_jitted)
    ctx.stop()

  @parameterized.named_parameters([
      ('plain_pmap_but_jit', True, True),
      ('plain_pmap', True, False),
      ('faked_pmap_but_jit', False, True),
      ('faked_pmap', False, False),
  ])
  def test_fake_pmap_(self, is_pmapped, jit_result):
    enable_patching = not is_pmapped

    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    # Call with context manager
    with fake.fake_pmap(enable_patching=enable_patching, jit_result=jit_result):
      _assert_pmapped(foo, fn_input, is_pmapped, jit_result)

    # Call with start/stop
    ctx = fake.fake_pmap(enable_patching=enable_patching, jit_result=jit_result)
    ctx.start()
    _assert_pmapped(foo, fn_input, is_pmapped, jit_result)
    ctx.stop()

  def test_fake_pmap_axis_name(self):

    with fake.fake_pmap():
      @jax.partial(jax.pmap, axis_name='i')
      @jax.partial(jax.pmap, axis_name='j')
      def f(_):
        return jax.lax.axis_index('i'), jax.lax.axis_index('j')
      x, y = f(jnp.zeros((4, 2)))

    self.assertEqual(x.tolist(), [[0, 0], [1, 1], [2, 2], [3, 3]])
    self.assertEqual(y.tolist(), [[0, 1], [0, 1], [0, 1], [0, 1]])

  @parameterized.named_parameters([
      ('fake_nothing', {
          'enable_pmap_patching': False,
          'enable_jit_patching': False
      }, True, True),
      ('fake_pmap', {
          'enable_pmap_patching': True,
          'enable_jit_patching': False
      }, False, True),
      # Default pmap will implicitly compile the function
      ('fake_jit', {
          'enable_pmap_patching': False,
          'enable_jit_patching': True
      }, True, False),
      ('fake_both', {
          'enable_pmap_patching': True,
          'enable_jit_patching': True
      }, False, False),
  ])
  def test_pmap_and_jit(self, fake_kwargs, is_pmapped, is_jitted):
    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    # Call with context manager
    with fake.fake_pmap_and_jit(**fake_kwargs):
      _assert_pmapped(foo, fn_input, is_pmapped)
      _assert_jitted(foo, fn_input, is_jitted)

    # Call with start/stop
    ctx = fake.fake_pmap_and_jit(**fake_kwargs)
    ctx.start()
    _assert_pmapped(foo, fn_input, is_pmapped)
    _assert_jitted(foo, fn_input, is_jitted)
    ctx.stop()

  @parameterized.named_parameters([
      ('fake_nothing', False, False),
      ('fake_pmap', True, False),
      ('fake_jit', False, True),
      ('fake_both', True, True),
  ])
  def test_with_kwargs(self, fake_pmap, fake_jit):
    with fake.fake_pmap_and_jit(fake_pmap, fake_jit):
      num_devices = len(jax.devices())

      @functools.partial(jax.pmap, axis_size=num_devices)
      @jax.jit
      def foo(x, y):
        return (x * 2) + y

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      expected = jnp.broadcast_to(jnp.array([3, 6]), (num_devices, 2))

      asserts.assert_trees_all_close(foo(x=inputs, y=inputs), expected)

  @parameterized.named_parameters([
      ('fake_nothing', False, 1),
      ('fake_pmap', True, 1),
      ('fake_nothing_no_static_args', False, ()),
      ('fake_pmap_no_static_args', True, ()),
  ])
  def test_with_static_broadcasted_argnums(self, fake_pmap, static_argnums):
    with fake.fake_pmap_and_jit(fake_pmap, enable_jit_patching=False):
      num_devices = len(jax.devices())

      # Note: mode='bar' is intended to test that we correctly handle kwargs
      # with defaults for which we don't pass a value at call time.
      @functools.partial(jax.pmap,
                         axis_size=num_devices,
                         static_broadcasted_argnums=static_argnums)
      @jax.jit
      def foo(x, multiplier, y, mode='bar'):
        if mode == 'bar':
          return (x * multiplier) + y
        else:
          return x

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      func = lambda: foo(inputs, 100, inputs)   # Pass multiplier=100.

      if static_argnums == 1:  # Should work.
        expected = jnp.broadcast_to(jnp.array([101, 202]), (num_devices, 2))
        result = func()
        asserts.assert_trees_all_close(result, expected)
      else:  # Should error.
        with self.assertRaises(ValueError):
          result = func()

  @parameterized.named_parameters([
      ('fake_nothing', False, False),
      ('fake_pmap', True, False),
      ('fake_jit', False, True),
      ('fake_both', True, True),
  ])
  def test_with_partial(self, fake_pmap, fake_jit):
    with fake.fake_pmap_and_jit(fake_pmap, fake_jit):
      num_devices = len(jax.devices())

      # Testing a common use-case where non-parallel arguments are partially
      # applied before pmapping
      def foo(x, y, flag):
        return (x * 2) + y if flag else (x + y)
      foo = functools.partial(foo, flag=True)

      foo = jax.pmap(foo, axis_size=num_devices)
      foo = jax.jit(foo)

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      expected = jnp.broadcast_to(jnp.array([3, 6]), (num_devices, 2))

      asserts.assert_trees_all_close(foo(inputs, inputs), expected)
      asserts.assert_trees_all_close(foo(x=inputs, y=inputs), expected)

  @parameterized.named_parameters([
      ('fake_nothing', False, False),
      ('fake_pmap', True, False),
      ('fake_jit', False, True),
      ('fake_both', True, True),
  ])
  def test_with_default_params(self, fake_pmap, fake_jit):
    with fake.fake_pmap_and_jit(fake_pmap, fake_jit):
      num_devices = len(jax.devices())

      # Default flag specified at definition time
      def foo(x, y, flag=True):
        return (x * 2) + y if flag else (x + y)

      default_foo = jax.pmap(foo, axis_size=num_devices)
      default_foo = jax.jit(default_foo)

      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      expected = jnp.broadcast_to(jnp.array([3, 6]), (num_devices, 2))
      asserts.assert_trees_all_close(default_foo(inputs, inputs), expected)
      asserts.assert_trees_all_close(default_foo(x=inputs, y=inputs), expected)

      # Default overriden by partial to execute other branch
      overidden_foo = functools.partial(foo, flag=False)
      overidden_foo = jax.pmap(overidden_foo, axis_size=num_devices)
      overidden_foo = jax.jit(overidden_foo)

      expected = jnp.broadcast_to(jnp.array([2, 4]), (num_devices, 2))
      asserts.assert_trees_all_close(overidden_foo(inputs, inputs), expected)
      asserts.assert_trees_all_close(
          overidden_foo(x=inputs, y=inputs), expected)


if __name__ == '__main__':
  absltest.main()
