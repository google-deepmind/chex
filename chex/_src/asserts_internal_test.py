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
import re

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts_internal as ai
from chex._src import pytypes
from chex._src import variants
import jax
import jax.numpy as jnp
import numpy as np


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
  def test_format_static_assertion(self, include_default_msg,
                                   include_custom_msg, exc_type):

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

  @variants.variants(with_jit=True, with_pmap=True)
  @parameterized.product(
      include_default_msg=(False, True),
      include_custom_msg=(False, True),
      exc_type=(AssertionError, ValueError),
  )
  def test_format_value_assertion(self, include_default_msg, include_custom_msg,
                                  exc_type):
    # Define a chex assertion.
    exc_msg = lambda x: f'{x} is non-positive.'

    def assert_positive(x):
      if x <= 0:
        raise AssertionError(exc_msg(x))

    def jittable_assert_positive(x):
      return (x > 0).all()

    chex_assert_positive = ai.chex_assertion(assert_positive,
                                             jittable_assert_positive)

    # Define a function with custom messages.
    n_steps = 3
    custom_msg = 'failed at iter {}' if include_custom_msg else ''
    n_traces = 0

    def pure_fn(x_pos, x_neg, start_from):
      nonlocal n_traces
      n_traces += 1

      y = x_pos
      # Run in a loop to generate different custom messages.
      for i in range(n_steps):
        step = start_from + i
        # Raises exception #1: `expected_exc_msg(x_neg, i)`.
        chex_assert_positive(  # pylint:disable=unexpected-keyword-arg
            x_neg,
            custom_message=custom_msg,
            custom_message_format_vars=[step],  # `step` is traced
            include_default_message=include_default_msg,
            exception_type=exc_type)

        # Passes.
        chex_assert_positive(x_pos)

        # Raises exception #2: `expected_exc_msg(3 * x_neg, custom_msg)`.
        chex_assert_positive(  # pylint:disable=unexpected-keyword-arg
            3 * x_neg,
            custom_message=custom_msg,
            custom_message_format_vars=[step],  # `step` is traced
            include_default_message=include_default_msg,
            exception_type=exc_type)
        y += x_pos + x_neg + i
      return y

    fn = ai.with_value_assertions(pure_fn, self.variant)

    # Check the format of raised exceptions' messages.
    def expected_exc_msg(x, custom_msg, device):
      msg = exc_msg(x) if include_default_msg else ''
      msg = rf'.*{msg} \[{custom_msg}\]' if custom_msg else msg
      return msg + f'.*{re.escape(str(device))}'

    # Check that the error is reported for every participating device.
    if self.variant is variants.ChexVariantType.WITH_PMAP:
      devices = jax.local_devices()
    else:
      devices = jax.local_devices()[0:1]

    # Check that 2 exceptions are raise at _every_ step (out of `n_steps`).
    # Value assertions are merged into one AssertionError, so we call the same
    # function multiple times but check presence of different a-n messages.
    for i in range(n_steps):
      start_from = 2 + i
      step_msg = custom_msg.format(start_from + i)
      x_pos = 3 * i
      x_neg = -2 * (i + 1)

      for device in devices:
        # Exception #1.
        with self.assertRaisesRegex(
            AssertionError,  # Value assertions raise `AssertionError`s.
            ai.get_err_regex(expected_exc_msg(x_neg, step_msg, device))):
          fn(x_pos, x_neg, start_from)

        # Exception #2.
        with self.assertRaisesRegex(
            AssertionError,  # Value assertions raise `AssertionError`s.
            ai.get_err_regex(expected_exc_msg(3 * x_neg, step_msg, device))):
          fn(x_pos, x_neg, start_from)

    self.assertEqual(n_traces, 1)


def _assert_trees_equal(tree_1: pytypes.ArrayTree,
                        tree_2: pytypes.ArrayTree) -> None:

  def _assert_fn(x1, x2):
    np.testing.assert_array_equal(
        ai.jnp_to_np_array(x1), ai.jnp_to_np_array(x2))

  jax.tree_map(_assert_fn, tree_1, tree_2)


class ChexAssertionsTest(variants.TestCase):
  """Tests for Chex assertions."""

  def setUp(self):
    super().setUp()

    def assert_tree_isfinite(tree):
      # Use jnp instead of np for testing purposes.
      if not all(jnp.isfinite(x).all() for x in jax.tree_leaves(tree)):
        raise AssertionError('Tree contains NaNs!')

    def jittable_assert_tree_isfinite(tree):
      # must be jittable
      return jnp.all(
          jnp.array([jnp.isfinite(x).all() for x in jax.tree_leaves(tree)]))

    self.chex_assert_isfinite_without_value = ai.chex_assertion(
        assert_tree_isfinite)

    self.chex_assert_isfinite_with_value = ai.chex_assertion(
        assert_tree_isfinite, jittable_assert_tree_isfinite)

  @variants.variants(with_jit=True, with_pmap=True)
  def test_value_assertion(self):
    eps = 1e-7

    def _make_log_abs_fn(assert_input_fn: ai.TChexAssertion):

      def _pure_log_fn(tree_1, tree_2):
        # Call twice to make sure all deps are retained after XLA optimizations.
        assert_input_fn(tree_1)
        assert_input_fn(tree_2)

        return jax.tree_map(lambda x1, x2: jnp.log(jnp.abs(x1 + x2) + eps),
                            tree_1, tree_2)

      return _pure_log_fn

    # Construct 3 versions of `log(|x1 + x2|)` function.
    log_abs_fn_no_assert = _make_log_abs_fn(lambda x: None)
    log_abs_fn_value_assert = _make_log_abs_fn(
        self.chex_assert_isfinite_with_value)
    log_abs_fn_static_assert = _make_log_abs_fn(
        self.chex_assert_isfinite_without_value)

    # Define correct and incorrect inputs.
    x = {'a': np.zeros((2,)), 'b': {'c': np.array([1, 2])}}
    x_with_nan = {'a': np.zeros((2,)), 'b': {'c': np.array([1, np.nan])}}

    # Test all versions return the same outputs.
    _assert_trees_equal(
        log_abs_fn_no_assert(x, x), log_abs_fn_static_assert(x, x))
    _assert_trees_equal(
        log_abs_fn_no_assert(x, x), log_abs_fn_value_assert(x, x))

    # Test on-device assertion.
    _assert_trees_equal(
        log_abs_fn_no_assert(x, x),
        ai.with_value_assertions(log_abs_fn_value_assert, self.variant)(x, x))

    # `ConcretizationTypeError` when static assertion is used in value checks.
    with self.assertRaisesRegex(
        jax.errors.ConcretizationTypeError,
        'Abstract tracer value encountered where concrete value is expected'):
      self.variant(log_abs_fn_static_assert)(x, x)
    with self.assertRaisesRegex(
        jax.errors.ConcretizationTypeError,
        'Abstract tracer value encountered where concrete value is expected'):
      self.variant(log_abs_fn_static_assert)(x, x_with_nan)

    # On-device assertion passes.
    _assert_trees_equal(
        log_abs_fn_no_assert(x, x),
        ai.with_value_assertions(log_abs_fn_value_assert, self.variant)(x, x))

    # Static assertion fails on incorrect inputs (without transformations).
    with self.assertRaisesRegex(AssertionError, 'Tree contains NaNs!'):
      log_abs_fn_static_assert(x_with_nan, x)

    with self.assertRaisesRegex(AssertionError, 'Tree contains NaNs!'):
      log_abs_fn_static_assert(x, x_with_nan)

    # Value assertion fails on incorrect inputs (with transformations).
    transformed_log_abs_fn_value_assert = ai.with_value_assertions(
        log_abs_fn_value_assert, self.variant)

    # Check that the error is reported for every participating device.
    if self.variant is variants.ChexVariantType.WITH_PMAP:
      devices = jax.local_devices()
    else:
      devices = jax.local_devices()[0:1]
    for device in devices:
      message_regexp = f'Tree contains NaNs!.*{re.escape(str(device))}'
      with self.assertRaisesRegex(AssertionError, message_regexp):
        transformed_log_abs_fn_value_assert(x, x_with_nan)
      with self.assertRaisesRegex(AssertionError, message_regexp):
        transformed_log_abs_fn_value_assert(x_with_nan, x)
      with self.assertRaisesRegex(AssertionError, message_regexp):
        transformed_log_abs_fn_value_assert(x_with_nan, x_with_nan)

    # Reports incorrect usage without `with_value_assertions()`.
    with self.assertRaisesRegex(
        RuntimeError,
        'can only be called from functions wrapped .*with_value_assertions'):
      self.variant(log_abs_fn_value_assert)(x, x_with_nan)

  @variants.variants(with_jit=True, without_jit=True)
  def test_static_assertion(self):
    # Tests that static assertions can be used without `with_value_assertions`.
    shape = (2, 3)

    # Define a simple static assertion.
    @ai.chex_assertion
    def chex_assert_shape(array, expected):
      if array.shape != expected:
        raise AssertionError('Wrong shape!')

    # Define a simple function that uses the assertion.
    def _sum_fn(tree):
      jax.tree_map(lambda x: chex_assert_shape(x, shape), tree)
      return sum(x.sum() for x in jax.tree_leaves(tree))

    # Passes in all contexts.
    x = {'a': np.ones(shape), 'b': {'c': np.ones(shape)}}
    self.assertEqual(_sum_fn(x), 2 * np.prod(shape))
    self.assertEqual(self.variant(_sum_fn)(x), 2 * np.prod(shape))
    self.assertEqual(
        ai.with_value_assertions(_sum_fn, self.variant)(x), 2 * np.prod(shape))

    # Fails in all contexts.
    x_wrong_shape = {'a': np.ones(shape), 'b': {'c': np.ones(shape + shape)}}
    with self.assertRaisesRegex(AssertionError, 'Wrong shape!'):
      _sum_fn(x_wrong_shape)
    with self.assertRaisesRegex(AssertionError, 'Wrong shape!'):
      self.variant(_sum_fn)(x_wrong_shape)
    with self.assertRaisesRegex(AssertionError, 'Wrong shape!'):
      ai.with_value_assertions(_sum_fn, self.variant)(x_wrong_shape)

  def test_jitted_fn_to_transform_fail(self):
    with self.assertRaisesRegex(ValueError,
                                'must not wrap JAX-transformed functions'):
      ai.with_value_assertions(jax.jit(lambda x: x.sum()), jax.pmap)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
