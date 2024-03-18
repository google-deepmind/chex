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
"""Tests for `warnings.py`."""

import functools

from absl.testing import absltest

from chex._src import warnings


@functools.partial(warnings.warn_only_n_pos_args_in_future, n=1)
def f(a, b, c):
  return a + b + c


@functools.partial(warnings.warn_deprecated_function, replacement='h')
def g(a, b, c):
  return a + b + c


class WarningsTest(absltest.TestCase):

  def test_warn_only_n_pos_args_in_future(self):
    with self.assertWarns(Warning):
      f(1, 2, 3)
    with self.assertWarns(Warning):
      f(1, 2, c=3)

  def test_warn_deprecated_function(self):
    with self.assertWarns(Warning):
      g(1, 2, 3)


if __name__ == '__main__':
  absltest.main()
