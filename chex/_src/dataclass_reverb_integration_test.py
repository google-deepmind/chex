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
"""Integration tests for `dataclass.py`.

These tests verify that chex dataclasses work with dm_tree and Reverb. These
were added to ensure compatibility if the dataclass implementation
changes, e.g. as a part of the proposed common chex/flax dataclass project.
"""

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import dataclass
from chex._src import pytypes

import jax
import numpy as np
import reverb
import tensorflow.compat.v2 as tf


chex_dataclass = dataclass.dataclass


@chex_dataclass(mappable_dataclass=True)
class Dataclass:
  a: pytypes.ArrayDevice
  b: int


class ReverbTest(parameterized.TestCase):
  """Test that dataclasses work with Reverb."""

  def setUp(self):
    super().setUp()
    tables = [
        reverb.Table(
            name='test',
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=100,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
    ]

    self.server = reverb.Server(tables, port=None)
    self.server_address = f'localhost:{self.server.port}'
    self.client = reverb.Client(self.server_address)

  def tearDown(self):
    super().tearDown()
    self.server.stop()
    del self.server

  def test_reverb_insert_sample(self):
    foo = Dataclass(a=np.ones(shape=(4, 4)), b=3)
    self.client.insert(foo, priorities={'test': 1.0})
    self.client.insert(foo, priorities={'test': 1.0})

    sample = list(self.client.sample('test', num_samples=1))
    self.assertLen(sample, 1)
    self.assertLen(sample[0], 1)
    sample = sample[0][0]

    bar = Dataclass.from_tuple(sample.data)
    asserts.assert_tree_all_close(foo, bar)

  def test_reverb_fails_on_non_mappable_dataclass(self):
    @chex_dataclass(mappable_dataclass=False)
    class NonMappableDataclass:
      a: pytypes.ArrayDevice

    foo = NonMappableDataclass(a=np.ones(shape=(4, 4)))
    with self.assertRaises(TypeError):
      self.client.insert(foo, priorities={'test': 1.0})

  def test_reverb_as_tf_dataset(self):
    foo = Dataclass(a=tf.ones(shape=(4,), dtype=tf.float32), b=2)

    # Write a trajectory
    with self.client.trajectory_writer(num_keep_alive_refs=5) as writer:
      for _ in range(5):
        writer.append(foo)

      writer.create_item(
          table='test',
          priority=1.0,
          trajectory={key: writer.history[key][:] for key in foo.keys()})
      writer.end_episode(timeout_ms=1000)

    # Sample a trajectory
    trajectory = Dataclass(
        a=tf.ones(shape=(5, 4)), b=tf.ones(shape=(5,), dtype=tf.int64) * 2)
    dataset = reverb.TrajectoryDataset(
        self.server_address,
        table='test',
        shapes=jax.tree_map(lambda x: x.shape, trajectory),
        dtypes=jax.tree_map(lambda x: x.dtype, trajectory),
        max_in_flight_samples_per_worker=10,
        rate_limiter_timeout_ms=10)

    for sample in dataset.take(1):
      sample = Dataclass.from_tuple(sample.data)
      asserts.assert_tree_all_close([trajectory, sample])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
