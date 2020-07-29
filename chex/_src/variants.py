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
"""Chex variants utilities."""

import functools
import inspect
import itertools

from absl import flags
from absl.testing import parameterized
from chex._src import fake
from chex._src import pytypes

import jax
from jax import tree_util
import jax.numpy as jnp
import toolz


FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "chex_skip_pmap_variant_if_single_device", True,
    "Whether to skip pmap variant if only one device is available.")


# `@chex.variants` returns a generator producing one test per variant.
# Therefore, users' TestCase class must support dynamic unrolling of these
# generators during module import. It is implemented and well-tested in
# `parameterized.TestCase`, hence we alias it as `variants.TestCase`.
#
# We choose to subclass instead of a simple alias, as Python doesn't allow
# multiple inheritance from the same class, and users may want to subclass their
# tests from both `chex.TestCase` and `parameterized.TestCase`.
#
# User is free to use any base class that supports generators unrolling
# instead of `variants.TestCase` or `parameterized.TestCase`. If a base class
# doesn't support this feature variant test fails with a corresponding error.
class TestCase(parameterized.TestCase):
  pass


tree_map = tree_util.tree_map


def count_num_calls(fn):
  """Counts number of function calls."""
  num_calls = 0

  @functools.wraps(fn)
  def fn_wrapped(*args, **kwargs):
    nonlocal num_calls
    num_calls += 1
    return fn(*args, **kwargs)

  return fn_wrapped, lambda: num_calls


class VariantsTestCaseGenerator:
  """TestCase generator for chex variants. Supports sharding."""

  def __init__(self, test_object, which_variants):
    self._which_variants = which_variants
    self._generated_names_freq = {}
    if hasattr(test_object, "__iter__"):
      # `test_object` is a generator (e.g. parameterised test).
      self._test_methods = [m for m in test_object]
    else:
      # `test_object` is a single test method.
      self._test_methods = [test_object]

  def add_variants(self, which_variants):
    """Merge variants."""
    for var, incl in which_variants.items():
      self._which_variants[var] = self._which_variants.get(var, False) or incl

  @property
  def __name__(self):
    msg = ("A test wrapper attempts to access __name__ of "
           "VariantsTestCaseGenerator. Usually, this happens when "
           "@parameterized wraps @variants.variants. Make sure that the "
           "@variants.variants wrapper is an outer one, i.e. nothing wraps it.")
    raise RuntimeError(msg)

  def __call__(self):
    msg = ("A test wrapper attempts to invoke __call__ of "
           "VariantsTestCaseGenerator: make sure that all `TestCase` instances "
           "that use variants inherit from `chex.TestCase`.")
    raise RuntimeError(msg)

  def _set_test_name(self, test_method, variant):
    """Set a name for the generated test."""
    name = getattr(test_method, "__name__", "")
    params_repr = getattr(test_method, "__x_params_repr__", "")
    chex_suffix = f"(chex variant == `{variant}`)"

    candidate_name = " ".join(filter(None, [name, params_repr, chex_suffix]))
    name_freq = self._generated_names_freq.get(candidate_name, 0)
    if name_freq:
      # Ensure that test names are unique.
      new_name = name + "_" + str(name_freq)
      unique_name = " ".join(filter(None, [new_name, params_repr, chex_suffix]))
    else:
      unique_name = candidate_name
    self._generated_names_freq[candidate_name] = name_freq + 1

    # Always use name for compatibility with `absl.testing.parameterized`.
    setattr(test_method, "__name__", unique_name)
    setattr(test_method, "__x_params_repr__", "")
    setattr(test_method, "__x_use_name__", True)
    return test_method

  def _inner_iter(self, test_method):
    """Generate chex variants for a single test."""

    def make_test(variant):

      @functools.wraps(test_method)
      def test(self, *args, **kwargs):
        # Skip pmap variant if only one device is available.

        if (variant == "with_pmap" and
            FLAGS.chex_skip_pmap_variant_if_single_device and
            jax.device_count() < 2):
          self.skipTest(f"Only 1 device is available ({jax.devices()}).")
          raise RuntimeError("This line should not be executed.")

        # n_cpu_devices assert.
        if FLAGS.chex_assert_multiple_cpu_devices:
          required_n_cpus = fake.get_n_cpu_devices_from_xla_flags()
          if required_n_cpus < 2:
            raise RuntimeError(
                f"Required number of CPU devices is {required_n_cpus} < 2."
                "Consider setting up your test module to use multiple CPU "
                " devices (see README.md) or disabling "
                "`chex_assert_multiple_cpu_devices` flag.")
          available_n_cpus = jax.device_count("cpu")
          if required_n_cpus != available_n_cpus:
            raise RuntimeError(
                "Number of available CPU devices is not equal to the required: "
                f"{available_n_cpus} != {required_n_cpus}")

        # Set up the variant.
        self.variant, num_calls = count_num_calls(_variant_decorators[variant])
        res = test_method(self, *args, **kwargs)
        if num_calls() == 0:
          raise RuntimeError(
              "Test is wrapped in @chex.variants, but never calls self.variant."
              " Consider debugging the test or removing @chex.variants wrapper."
              f" (variant: {variant})")
        return res

      self._set_test_name(test, variant)
      return test

    return (make_test(var_name)
            for var_name, is_included in self._which_variants.items()
            if is_included)

  def __iter__(self):
    """Generate chex variants for each test case."""
    return itertools.chain(*(self._inner_iter(m) for m in self._test_methods))


@toolz.curry
def _variants_fn(test_object, **which_variants):
  """Implements `variants` and `all_variants`."""

  if isinstance(test_object, VariantsTestCaseGenerator):
    # Merge variants for nested wrappers.
    test_object.add_variants(which_variants)
  else:
    test_object = VariantsTestCaseGenerator(test_object, which_variants)

  return test_object


@toolz.curry
def variants(test_method,
             with_jit: bool = False,
             without_jit: bool = False,
             with_device: bool = False,
             without_device: bool = False,
             with_pmap: bool = False):
  """Decorates a test to expose Chex variants.

  The decorated test has access to a decorator called `self.variant`, which
  may be applied to functions to test different JAX behaviors. Consider:

  ```python
  @chex.variants(with_jit=True, without_jit=True)
  def test(self):
    @self.variant
    def f(x, y):
      return x + y

    self.assertEqual(f(1, 2), 3)
  ```

  In this example, the function `test` will be called twice: once with `f`
  jitted (i.e. using `jax.jit`) and another where `f` is not jitted.

  Variants `with_jit=True` and `with_pmap=True` accept additional specific to
  them arguments. Example:
  ```python
  @chex.variants(with_jit=True)
  def test(self):
    @self.variant(static_argnums=(1,))
    def f(x, y):
      # `y` is not traced.
      return x + y

    self.assertEqual(f(1, 2), 3)
  ```

  Variant `with_pmap=True` also accepts `broadcast_args_to_devices`
  (whether to broadcast each input argument to all participating devices),
  `reduce_fn` (a function to apply to results of pmapped `fn`), and
  `n_devices` (number of devices to use in the `pmap` computation).
  See the docstring of `_with_pmap` for more details (including default values).

  If used with `absl.testing.parameterized`, @chex.variants must wrap it:
  ```python
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters('test', *args)
  def test(self, *args):
    ...
  ```

  Tests that use this wrapper must be inherited from `parameterized.TestCase`.
  For more examples see 'variants_test.py'.

  Args:
    test_method: Test method to decorate.
    with_jit: Whether to test with `jax.jit`.
    without_jit: Whether to test without `jax.jit`. Any jit compilation done
      within the test method will not be affected.
    with_device: Whether to test with args placed on device, using
      `jax.device_put`.
    without_device: Whether to test with args (explicitly) not placed on device,
      using `jax.device_get`.
    with_pmap: Whether to test with `jax.pmap`, with computation duplicated
      across devices.

  Returns:
    Decorated test_method.
  """
  return _variants_fn(
      test_method,
      with_jit=with_jit,
      without_jit=without_jit,
      with_device=with_device,
      without_device=without_device,
      with_pmap=with_pmap)


@toolz.curry
def all_variants(test_method,
                 with_jit: bool = True,
                 without_jit: bool = True,
                 with_device: bool = True,
                 without_device: bool = True,
                 with_pmap: bool = True):
  """Equivalent to `variants` but with flipped defaults."""
  return _variants_fn(
      test_method,
      with_jit=with_jit,
      without_jit=without_jit,
      with_device=with_device,
      without_device=without_device,
      with_pmap=with_pmap)


def check_variant_arguments(variant_fn):
  """Raises `ValueError` if `variant_fn` got an unknown argument."""

  @functools.wraps(variant_fn)
  def wrapper(*args, **kwargs):
    unknown_args = set(kwargs.keys()) - _valid_kwargs_keys
    if unknown_args:
      raise ValueError(f"Unknown arguments in `self.variant`: {unknown_args}.")
    return variant_fn(*args, **kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _with_jit(fn,
              static_argnums=(),
              device=None,
              backend=None,
              **unused_kwargs):
  """Variant that applies `jax.jit` to fn."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return jax.jit(fn, static_argnums, device, backend)(*args, **kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _without_jit(fn, **unused_kwargs):
  """Variant that does not apply `jax.jit` to a fn (identity)."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return fn(*args, **kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _with_device(fn, ignore_argnums=(), **unused_kwargs):
  """Variant that applies `jax.device_put` to the args of fn."""

  if isinstance(ignore_argnums, int):
    ignore_argnums = (ignore_argnums,)

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):

    def put(x):
      try:
        return jax.device_put(x)
      except TypeError:  # not a valid JAX type
        return x

    device_args = [
        arg if idx in ignore_argnums else tree_map(put, arg)
        for idx, arg in enumerate(args)
    ]
    device_kwargs = tree_map(put, kwargs)
    return fn(*device_args, **device_kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _without_device(fn, **unused_kwargs):
  """Variant that applies `jax.device_get` to the args of fn."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):

    def get(x):
      if isinstance(x, jnp.DeviceArray):
        return jax.device_get(x)
      return x

    no_device_args = tree_map(get, args)
    no_device_kwargs = tree_map(get, kwargs)
    return fn(*no_device_args, **no_device_kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _with_pmap(fn,
               broadcast_args_to_devices=True,
               reduce_fn="first_device_output",
               n_devices=None,
               axis_name="i",
               devices=None,
               in_axes=0,
               static_broadcasted_argnums=(),
               static_argnums=(),
               backend=None,
               **unused_kwargs):
  """Variant that applies `jax.pmap` to fn.

  Args:
    fn: a function to wrap.
    broadcast_args_to_devices: whether to broadcast `fn` args to pmap format
      (i.e. pmapped axes' sizes == a number of devices).
    reduce_fn: a function to apply to outputs of `fn`.
    n_devices: a number of devices to use (can specify a `backend` if required).
    axis_name: passed to `pmap`.
    devices: passed to `pmap`.
    in_axes: passed to `pmap`.
    static_broadcasted_argnums: passed to `pmap`.
    static_argnums: alias of static_broadcasted_argnums.
    backend: passed to `pmap`.
    **unused_kwargs: unused kwargs (e.g. related to other variants).

  Returns:
    Wrapped `fn` that accepts `args` and `kwargs` and returns a superposition of
    `reduce_fn` and `fn` applied to them.

  Raises:
    `ValueError` if `broadcast_args_to_devices` used with `in_axes` or
                   `static_broadcasted_argnums`;
                 if number of available devices is less than a required;
                 pmappable arg axes' sizes are not equal to a number of devices.
  """

  if broadcast_args_to_devices and in_axes != 0:
    raise ValueError(
        "Do not use `broadcast_args_to_devices` when specifying `in_axes`.")

  # Set up a reduce function.
  if reduce_fn == "first_device_output":
    reduce_fn = lambda t: tree_map(lambda x: x[0], t)
  elif reduce_fn == "identity" or reduce_fn is None:  # Identity.
    reduce_fn = lambda t: t

  if not static_argnums and static_argnums != 0:
    static_argnums = static_broadcasted_argnums
  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)

  @functools.wraps(fn)
  def wrapper(*args: pytypes.ArrayTree, **kwargs: pytypes.ArrayTree):
    if kwargs and (in_axes != 0 or static_argnums):
      raise ValueError("Do not use kwargs with `in_axes` or `static_argnums` "
                       "in pmapped function.")
    devices_ = list(devices or jax.devices(backend))
    n_devices_ = n_devices or len(devices_)
    devices_ = devices_[:n_devices_]
    if len(devices_) != n_devices_:
      raise ValueError("Number of available devices is less than required for "
                       f"test ({len(devices_)} < {n_devices_})")

    bcast_fn = lambda x: jnp.broadcast_to(x, (n_devices_,) + jnp.array(x).shape)
    if broadcast_args_to_devices:
      args = [
          tree_map(bcast_fn, arg) if idx not in static_argnums else arg
          for idx, arg in enumerate(args)
      ]
      kwargs = tree_map(bcast_fn, kwargs)
    else:
      # Pmappable axes size must be equal to number of devices.
      in_axes_ = in_axes if isinstance(in_axes,
                                       (tuple, list)) else [in_axes] * len(args)
      is_pmappable_arg = [
          idx not in static_argnums and in_axes_[idx] is not None
          for idx in range(len(args))
      ]
      for is_pmappable_arg, arg in zip(is_pmappable_arg, args):
        if not is_pmappable_arg:
          continue
        if not all(x.shape[0] == n_devices_ for x in jax.tree_leaves(arg)):
          shapes = tree_map(jnp.shape, arg)
          raise ValueError(
              f"Pmappable arg axes size must be equal to number of devices, "
              f"got: {shapes} (expected the first dim to be {n_devices_}). "
              "Consider setting `broadcast_args_to_devices=True`.")

    res = jax.pmap(
        fn,
        axis_name=axis_name,
        devices=devices_,
        in_axes=in_axes,
        static_broadcasted_argnums=static_argnums,
        backend=backend)(*args, **kwargs)

    return reduce_fn(res)

  return wrapper


_variant_decorators = {
    "with_jit": _with_jit,
    "without_jit": _without_jit,
    "with_device": _with_device,
    "without_device": _without_device,
    "with_pmap": _with_pmap,
}


# Collect valid argument names from all variant decorators.
_valid_kwargs_keys = set()
for fn_ in _variant_decorators.values():
  original_fn = fn_.func.__wrapped__
  _valid_kwargs_keys.update(inspect.getfullargspec(original_fn).args)
