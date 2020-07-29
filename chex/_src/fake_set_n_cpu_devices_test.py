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
"""Test for `set_n_cpu_devices` from `fake.py`.

This test is isolated to ensure hermeticity because its execution changes
XLA backend configuration.
"""

from absl.testing import absltest
from chex._src import asserts
from chex._src import fake


class DevicesSetterTest(absltest.TestCase):

  def test_set_n_cpu_devices(self):
    # Should not initialize backends.
    fake.set_n_cpu_devices(4)

    # Hence, this one does not fail.
    fake.set_n_cpu_devices(6)

    # This assert initializes backends.
    asserts.assert_devices_available(6, 'cpu', backend='cpu')

    # Which means that next call must fail.
    with self.assertRaisesRegex(RuntimeError,
                                'Attempted to set 8 devices, but 6 CPUs.+'):
      fake.set_n_cpu_devices(8)


if __name__ == '__main__':
  absltest.main()
