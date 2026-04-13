"""Tests for scalar assertions with vmap compatibility (Issue #389)."""

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import asserts_chexify
import jax
import jax.numpy as jnp


class ScalarAssertionsVmapTest(parameterized.TestCase):
  """Tests for vmap-compatible scalar assertions."""

  def test_assert_scalar_positive_with_vmap_mwe(self):
    """Test the exact MWE from issue #389."""
    x_scalar = 1.
    x_vector = jnp.array([1., 1.])
    
    # Test 1: Scalar (should work as before, without chexify)
    asserts.assert_scalar_positive(x_scalar)  # Works
    
    # Test 2: Vector with vmap (the requested feature)
    def test_vmap(x):
      jax.vmap(asserts.assert_scalar_positive)(x)
      return x
    
    test_vmap_chexified = asserts_chexify.chexify(jax.jit(test_vmap), async_check=False)
    
    # Should pass
    result = test_vmap_chexified(x_vector)
    self.assertTrue(jnp.array_equal(result, x_vector))

  def test_assert_scalar_positive_with_vmap_failure(self):
    """Test that vmap correctly detects negative values."""
    x_negative = jnp.array([1., -1., 3.])
    
    def check_positive(x):
      jax.vmap(asserts.assert_scalar_positive)(x)
      return x
    
    check_positive_chexified = asserts_chexify.chexify(jax.jit(check_positive), async_check=False)
    
    # Should fail
    with self.assertRaisesRegex(AssertionError, 'must be positive'):
      check_positive_chexified(x_negative)

  @parameterized.parameters(
      ('assert_scalar_positive', jnp.array([1., 2., 3.]), jnp.array([1., -1., 3.])),
      ('assert_scalar_non_negative', jnp.array([0., 1., 2.]), jnp.array([0., -1., 2.])),
      ('assert_scalar_negative', jnp.array([-1., -2., -3.]), jnp.array([-1., 1., -3.])),
  )
  def test_scalar_assertions_vmap(self, assertion_name, valid_input, invalid_input):
    """Test all scalar assertions with vmap."""
    assertion_fn = getattr(asserts, assertion_name)
    
    def check_fn(x):
      jax.vmap(assertion_fn)(x)
      return x
    
    check_fn_chexified = asserts_chexify.chexify(jax.jit(check_fn), async_check=False)
    
    # Valid input should pass
    result = check_fn_chexified(valid_input)
    self.assertTrue(jnp.array_equal(result, valid_input))
    
    # Invalid input should fail
    with self.assertRaises(AssertionError):
      check_fn_chexified(invalid_input)

  def test_backward_compatibility_without_chexify(self):
    """Test that assertions still work without chexify for scalars."""
    # These should all work as before
    asserts.assert_scalar_positive(1.0)
    asserts.assert_scalar_positive(5)
    asserts.assert_scalar_non_negative(0.0)
    asserts.assert_scalar_non_negative(1)
    asserts.assert_scalar_negative(-1.0)
    asserts.assert_scalar_negative(-5)
    
    # These should all fail
    with self.assertRaisesRegex(AssertionError, 'must be positive'):
      asserts.assert_scalar_positive(-1.0)
    
    with self.assertRaisesRegex(AssertionError, 'must be non-negative'):
      asserts.assert_scalar_non_negative(-1.0)
    
    with self.assertRaisesRegex(AssertionError, 'must be negative'):
      asserts.assert_scalar_negative(1.0)

  def test_with_jit_only(self):
    """Test that assertions work with jit (without vmap)."""
    def check_positive(x):
      asserts.assert_scalar_positive(x)
      return x * 2
    
    check_positive_chexified = asserts_chexify.chexify(jax.jit(check_positive), async_check=False)
    
    # Should pass
    result = check_positive_chexified(5.0)
    self.assertEqual(result, 10.0)
    
    # Should fail
    with self.assertRaisesRegex(AssertionError, 'must be positive'):
      check_positive_chexified(-5.0)

  def test_with_pmap(self):
    """Test that assertions work with pmap."""
    devices = jax.local_devices()
    if len(devices) < 2:
      self.skipTest('Test requires at least 2 devices')
    
    def check_positive(x):
      asserts.assert_scalar_positive(x)
      return x * 2
    
    check_positive_pmapped = asserts_chexify.chexify(jax.pmap(check_positive), async_check=False)
    
    # Should pass
    x_valid = jnp.array([1.0, 2.0])
    result = check_positive_pmapped(x_valid)
    self.assertTrue(jnp.array_equal(result, x_valid * 2))
    
    # Should fail
    x_invalid = jnp.array([1.0, -2.0])
    with self.assertRaisesRegex(AssertionError, 'must be positive'):
      check_positive_pmapped(x_invalid)

  def test_nested_vmap(self):
    """Test that assertions work with nested vmap."""
    def check_positive(x):
      # x has shape (3, 4)
      jax.vmap(jax.vmap(asserts.assert_scalar_positive))(x)
      return x
    
    check_positive_chexified = asserts_chexify.chexify(jax.jit(check_positive), async_check=False)
    
    # Should pass
    x_valid = jnp.ones((3, 4))
    result = check_positive_chexified(x_valid)
    self.assertTrue(jnp.array_equal(result, x_valid))
    
    # Should fail
    x_invalid = jnp.array([[1., 2., 3., 4.],
                           [5., -6., 7., 8.],
                           [9., 10., 11., 12.]])
    with self.assertRaisesRegex(AssertionError, 'must be positive'):
      check_positive_chexified(x_invalid)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
