"""Example demonstrating how to use chex.variants with parametrization.

This example shows the correct pattern for combining `chex.variants` with parametrization.
Since `chex.variants` requires the test class to inherit from `chex.TestCase` (which uses `unittest.TestCase` logic),
native `pytest.mark.parametrize` DOES NOT work because `pytest` does not support parametrization
on `unittest.TestCase` subclasses.

Instead, you should use `absl.testing.parameterized` (which `chex` builds upon).
This works seamlessly when run via `pytest`.

To run these examples:
    pytest variants_pytest_example.py -v
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized


class TestBasicVariants(chex.TestCase):
    """Examples of basic variant usage without parametrization."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_simple_addition(self):
        """Test that runs twice: once with jit, once without."""

        @self.variant
        def add(x, y):
            return x + y

        result = add(2, 3)
        self.assertEqual(result, 5)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_jax_array_operations(self):
        """Test JAX array operations across multiple variants."""

        @self.variant
        def square_and_sum(arr):
            return jnp.sum(arr ** 2)

        arr = jnp.array([1.0, 2.0, 3.0])
        result = square_and_sum(arr)
        expected = 14.0  # 1^2 + 2^2 + 3^2

        np.testing.assert_allclose(result, expected)


class TestCombinedParametrize(chex.TestCase):
    """Examples combining chex.variants with parameterized.
    
    CRITICAL: @chex.variants MUST be the OUTSIDE decorator (applied first).
    """

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters([1, 2, 3])
    def test_basic_parametrize(self, n):
        """Test runs 6 times total (3 parameters × 2 variants)."""

        @self.variant
        def add_one(x):
            return x + 1

        result = add_one(n)
        self.assertEqual(result, n + 1)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        (1, 2, 3),
        (5, 7, 12),
        (-3, 3, 0),
        (0, 0, 0),
    )
    def test_multiple_parameters(self, x, y, expected):
        """Test with multiple parameters per test case."""

        @self.variant
        def add(a, b):
            return a + b

        result = add(x, y)
        self.assertEqual(result, expected)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small", [1, 2, 3], 6),
        ("medium", [10, 20, 30], 60),
        ("negative", [-1, -2, -3], -6),
    )
    def test_with_custom_ids(self, arr, expected_sum):
        """Demonstrate using custom test IDs/Names with named_parameters."""

        @self.variant
        def sum_array(x):
            return jnp.sum(jnp.array(x))

        result = sum_array(arr)
        self.assertEqual(float(result), expected_sum)


class TestMultipleVariants(chex.TestCase):
    """Examples using multiple variant types."""

    @chex.variants(
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
    )
    @parameterized.parameters([1, 2, 3])
    def test_four_variants(self, n):
        """Test runs 12 times total (3 parameters × 4 variants)."""

        @self.variant
        def square(x):
            return x * x

        result = square(jnp.array(n))
        expected = n * n
        np.testing.assert_allclose(result, expected)

    # all_variants enables ALL variant types by default
    @chex.all_variants(with_pmap=False)  # Exclude pmap for single-device
    @parameterized.parameters([1.0, 2.0, 3.0])
    def test_all_variants(self, value):
        """Use all_variants for comprehensive testing."""

        @self.variant
        def absolute_value(x):
            return jnp.abs(x)

        result = absolute_value(jnp.array(-value))
        self.assertEqual(float(result), value)


class TestJAXSpecificOperations(chex.TestCase):
    """Examples testing JAX-specific operations."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        ((2, 3), 5.0),
        ((4, 4), 10.0),
        ((1, 5), 2.0),
    )
    def test_array_creation(self, shape, fill_value):
        """Test JAX array creation across variants."""

        # 's' (shape) must be static for JIT compilation
        @self.variant(static_argnums=(0,))
        def create_filled_array(s, v):
            return jnp.full(s, v)

        result = create_filled_array(shape, fill_value)

        # Use chex assertions for array properties
        chex.assert_shape(result, shape)
        self.assertTrue(jnp.all(result == fill_value))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("axis_0", jnp.array([[1, 2], [3, 4]]), 0, (2,)),
        ("axis_1", jnp.array([[1, 2], [3, 4]]), 1, (2,)),
        ("multi_axis", jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (0, 1), (2,)),
    )
    def test_reduction_operations(self, input_array, axis, expected_shape):
        """Test reduction operations with different axes."""

        # 'ax' (axis) must be static for JIT compilation
        @self.variant(static_argnums=(1,))
        def sum_along_axis(arr, ax):
            return jnp.sum(arr, axis=ax)

        result = sum_along_axis(input_array, axis)
        chex.assert_shape(result, expected_shape)


class TestVariantSpecificArguments(chex.TestCase):
    """Examples showing variant-specific arguments (e.g., for jit)."""

    @chex.variants(with_jit=True)
    @parameterized.parameters([10, 20, 30])
    def test_static_argnums(self, multiplier):
        """Demonstrate passing static_argnums to jit variant."""

        # static_argnums makes the second argument non-traced in jit
        @self.variant(static_argnums=(1,))
        def multiply_by_static(x, factor):
            # factor won't be traced when jitted
            return x * factor

        arr = jnp.array([1.0, 2.0, 3.0])
        result = multiply_by_static(arr, multiplier)
        expected = arr * multiplier

        chex.assert_trees_all_close(result, expected)

    @chex.variants(with_jit=True)
    # Note: Named parameters can clarify complex test cases
    @parameterized.named_parameters(
        ("square", (3, 3)),
        ("wide", (2, 4)),
        ("tall", (5, 2)),
    )
    def test_static_argnames(self, shape):
        """Demonstrate using static_argnames for keyword arguments."""

        @self.variant(static_argnames=('output_shape',))
        def reshape_array(x, output_shape):
            return jnp.reshape(x, output_shape)

        size = shape[0] * shape[1]
        arr = jnp.arange(size)
        result = reshape_array(arr, output_shape=shape)

        chex.assert_shape(result, shape)
