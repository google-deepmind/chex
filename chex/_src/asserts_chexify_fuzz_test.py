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
"""Fuzz test for `asserts_chexify.py`."""

import concurrent.futures
import random
import time

from absl.testing import absltest
from chex._src import asserts
from chex._src import asserts_chexify
from chex._src import variants
import jax
import jax.numpy as jnp


class AssertsChexifyFuzzTest(variants.TestCase):
  """Fuzz test for thread safety of chexify."""

  def test_thread_safety(self):

    def assert_negative():
      result = jnp.ones(shape=())
      # This assert will always fail.
      asserts.assert_scalar_negative(result)
      return result

    def chexified_assert_negative():
      fn = asserts_chexify.chexify(assert_negative, async_check=True)
      fn()
      # Introduce random delay between the two calls, otherwise we will not
      # get interleaving of the two operations between threads because they
      # happen too quickly.
      time.sleep(random.uniform(0.01, 0.02))
      asserts_chexify.block_until_chexify_assertions_complete()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
      futures = []
      for _ in range(1000):
        future = executor.submit(chexified_assert_negative)
        futures.append(future)

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except AssertionError:
          pass

    asserts_chexify.block_until_chexify_assertions_complete()


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
