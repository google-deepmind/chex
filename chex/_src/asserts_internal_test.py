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
"""Tests for `asserts_internal.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts_internal as ai
from chex._src import variants
import jax


class IsTraceableTest(variants.TestCase):

  @variants.variants(with_jit=True, with_pmap=True)
  @parameterized.named_parameters(
      ('CPP_JIT', True),
      ('PY_JIT', False),
  )
  def test_is_traceable(self, cpp_jit):
    prev_state = jax.config.FLAGS.experimental_cpp_jit
    jax.config.FLAGS.experimental_cpp_jit = cpp_jit

    def dummy_wrapper(fn):

      @functools.wraps(fn)
      def fn_wrapped(fn, *args):
        return fn(args)

      return fn_wrapped

    fn = lambda x: x.sum()
    wrapped_fn = dummy_wrapper(fn)
    self.assertFalse(ai.is_traceable(fn))
    self.assertFalse(ai.is_traceable(wrapped_fn))

    var_fn = self.variant(fn)
    wrapped_var_f = dummy_wrapper(var_fn)
    var_wrapped_f = self.variant(wrapped_fn)
    self.assertTrue(ai.is_traceable(var_fn))
    self.assertTrue(ai.is_traceable(wrapped_var_f))
    self.assertTrue(ai.is_traceable(var_wrapped_f))

    jax.config.FLAGS.experimental_cpp_jit = prev_state


class ExceptionMessageFormatTest(variants.TestCase):

  @parameterized.product(
      include_default_msg=(False, True),
      include_custom_msg=(False, True),
      exc_type=(AssertionError, ValueError),
  )
  def test_format(self, include_default_msg, include_custom_msg, exc_type):

    exc_msg = lambda x: f'{x} is non-positive.'

    @ai.chex_assertion
    def assert_positive(x):
      if x <= 0:
        raise AssertionError(exc_msg(x))

    @ai.chex_assertion
    def assert_each_positive(*args):
      for x in args:
        assert_positive(x)

    # Pass.
    assert_positive(1)
    assert_each_positive(1, 2, 3)

    # Check the format of raised exceptions' messages.
    def expected_exc_msg(x, custom_msg):
      msg = exc_msg(x) if include_default_msg else ''
      msg = rf'{msg} \[{custom_msg}\]' if custom_msg else msg
      return msg

    # Run in a loop to generate different custom messages.
    for i in range(3):
      custom_msg = f'failed at iter {i}' if include_custom_msg else ''

      with self.assertRaisesRegex(
          exc_type, ai.get_err_regex(expected_exc_msg(-1, custom_msg))):
        assert_positive(  # pylint:disable=unexpected-keyword-arg
            -1,
            custom_message=custom_msg,
            include_default_message=include_default_msg,
            exception_type=exc_type)

      with self.assertRaisesRegex(
          exc_type, ai.get_err_regex(expected_exc_msg(-3, custom_msg))):
        assert_each_positive(  # pylint:disable=unexpected-keyword-arg
            1,
            -3,
            2,
            custom_message=custom_msg,
            include_default_message=include_default_msg,
            exception_type=exc_type)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
