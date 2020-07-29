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
"""Tests for `fake.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

from chex._src import fake
from chex._src import pytypes
import jax
import jax.numpy as jnp

ArrayBatched = pytypes.ArrayBatched
ArrayDevice = pytypes.ArrayDevice
ArraySharded = pytypes.ArraySharded


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule():
  fake.set_n_cpu_devices()


class PmapFakeTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('plain_jit', fake.fake_jit, False, 1),
      ('faked_jit', fake.fake_jit, True, 2),
  ])
  def test_fake_jit(self, context, patch, expected_execution_count):
    # We test whether the function has been jitted by introducing a counter
    # variable as a side-effect. When the function is repeatedly called, jitted
    # code will only execute the side-effect once
    python_execution_count = 0
    with context(patch):

      @jax.jit
      def foo(x):
        nonlocal python_execution_count
        python_execution_count += 1
        return x * 2

      foo(jnp.array([1, 2]))
      self.assertEqual(python_execution_count, 1)

      foo(jnp.array([1, 2]))
      self.assertEqual(python_execution_count, expected_execution_count)

  @parameterized.named_parameters([
      ('plain_pmap', fake.fake_pmap, False, ArraySharded),
      ('faked_pmap', fake.fake_pmap, True, ArrayDevice),
  ])
  def test_fake_pmap(self, context, patch, expected_type):
    # We test whether the function has been pmapped by inspecting the type of
    # the function output, if it is a sharded array type then the function has
    # been pmapped
    with context(patch):
      num_devices = len(jax.devices())

      @functools.partial(jax.pmap, axis_size=num_devices)
      def foo(x):
        return x * 2

      # pmap over all available devices
      x = jnp.array([1, 2])
      x = jnp.broadcast_to(x, (num_devices,) + x.shape)
      output = foo(x)
      self.assertEqual(type(output), expected_type)

  @parameterized.named_parameters([
      ('fake_nothing', fake.fake_pmap_and_jit, False, False, ArraySharded, 1),
      ('fake_pmap', fake.fake_pmap_and_jit, True, False, ArrayDevice, 1),
      # Default pmap will implicitly jit compile the function
      ('fake_jit', fake.fake_pmap_and_jit, False, True, ArraySharded, 1),
      ('fake_both', fake.fake_pmap_and_jit, True, True, ArrayDevice, 2),
  ])
  def test_pmap_and_jit(self, context, fake_pmap, fake_jit, expected_type,
                        expected_execution_count):
    python_execution_count = 0
    with context(fake_pmap, fake_jit):
      num_devices = len(jax.devices())
      @functools.partial(jax.pmap, axis_size=num_devices)
      @jax.jit
      def foo(x):
        nonlocal python_execution_count
        python_execution_count += 1
        return x * 2

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      output = foo(inputs)
      self.assertEqual(type(output), expected_type)
      self.assertEqual(python_execution_count, 1)

      foo(inputs)
      self.assertEqual(python_execution_count, expected_execution_count)


if __name__ == '__main__':
  absltest.main()
