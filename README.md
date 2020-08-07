# Chex

Chex is a library of utilities for helping to write reliable JAX code.

This includes utils to help:

* Instrument your code (e.g. assertions)
* Debug (e.g. transforming `pmaps` in `vmaps` within a context manager).
* Test JAX code across many `variants` (e.g. jitted vs non-jitted).

## Installation

Chex can be installed with pip directly from github, with the following command:

`pip install git+git://github.com/deepmind/chex.git`

or from PyPI:

`pip install chex`

## Modules Overview

### Assertions ([asserts.py](https://github.com/deepmind/chex/blob/master/chex/_src/asserts.py))

One limitation of PyType annotations for JAX is that they do not support the
specification of `DeviceArray` ranks, shapes or dtypes. Chex includes a number
of functions that allow flexible and concise specification of these properties.

E.g. suppose you want to ensure that all tensors `t1`, `t2`, `t3` have the same
shape, and that tensors `t4`, `t5` have rank `2` and (`3` or `4`), respectively.

```python
chex.assert_equal_shape([t1, t2, t3])
chex.assert_rank([t4, t5], [2, {3, 4}])
```

More examples:

```python
from chex import assert_shape, assert_rank, ...

assert_shape(x, (2, 3))                # x has shape (2, 3)
assert_shape([x, y], [(), (2,3)])      # x is scalar and y has shape (2, 3)

assert_rank(x, 0)                      # x is scalar
assert_rank([x, y], [0, 2])            # x is scalar and y is a rank-2 array
assert_rank([x, y], {0, 2})            # x and y are scalar OR rank-2 arrays

assert_type(x, int)                    # x has type `int` (x can be an array)
assert_type([x, y], [int, float])      # x has type `int` and y has type `float`

assert_equal_shape([x, y, z])          # x, y, and z have equal shapes

assert_tree_all_close(tree_x, tree_y)  # values and structure of trees match
assert_tree_all_finite(tree_x)         # all tree_x leaves are finite

assert_devices_available(2, 'gpu')     # 2 GPUs available
assert_tpu_available()                 # at least 1 TPU available

assert_numerical_grads(f, (x, y), j)   # f^{(j)}(x, y) matches numerical grads
```

See documentation of [asserts.py](https://github.com/deepmind/chex/blob/master/chex/_src/asserts.py) for details on all supported assertions.

### Dataclass ([dataclass.py](https://github.com/deepmind/chex/blob/master/chex/_src/dataclass.p))

JAX-friendly dataclass implementation reusing python [dataclasses](https://docs.python.org/3/library/dataclasses.html#module-dataclasses).

Chex implementation of `dataclass` registers dataclasses as internal [_PyTree_
nodes](https://jax.readthedocs.io/en/latest/pytrees.html) to ensure
compatibility with JAX data structures.

### Fakes ([fake.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake.py))

Debugging in JAX is made more difficult by code transformations such as `jit`
and `pmap`, which introduce optimizations that make code hard to inspect and
trace. It can also be difficult to disable those transformations during
debugging as they can be called at several places in the underlying
code. Chex provides tools to globally replace `jax.jit` with a no-op
transformation and `jax.pmap` with a (non-parallel) `jax.vmap`, in order to more
easily debug code in a single-device context.

For example, you can use Chex to fake `pmap` and have it replaced with a `vmap`.
This can be achieved by wrapping your code with a context manager:

```python
with chex.fake_pmap(enable_patching=True):
  @jax.pmap
  def fn(inputs):
    ...

  # Function will be vmapped over inputs
  fn(inputs)
```

The same functionality can also be invoked with `start` and `stop`:

```python
fake_pmap = chex.fake_pmap(enable_patching=True)
fake_pmap.start()
... your jax code ...
fake_pmap.stop()
```

In addition, you can fake a real multi-device test environment with a
multi-threaded CPU. See section **Faking multi-device test environments** for
more details.

See documentation in [fake.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake.py) and examples in [fake_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake_test.py) for more details.

### Test variants ([variants.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants.py))

JAX relies extensively on code transformation and compilation, meaning that it
can be hard to ensure that code is properly tested. For instance, just testing a
python function using JAX code will not cover the actual code path that is
executed when jitted, and that path will also differ whether the code is jitted
for CPU, GPU, or TPU. This has been a source of obscure and hard to catch bugs
where XLA changes would lead to undesirable behaviours that however only
manifest in one specific code transformation.

Variants make it easy to ensure that unit tests cover different ‘variations’ of
a function, by providing a simple decorator that can be used to repeat any test
under all (or a subset) of the relevant code transformations.

E.g. suppose you want to test the output of a function `fn` with or without jit.
You can use `chex.variants` to run the test with both the jitted and non-jitted
version of the function by simply decorating a test method with
`@chex.variants`, and then using `self.variant(fn)` in place of `fn` in the body
of the test.

```python
def fn(x, y):
  return x + y
...

class ExampleTest(chex.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test(self):
    var_fn = self.variant(fn)
    self.assertEqual(fn(1, 2), 3)
    self.assertEqual(var_fn(1, 2), fn(1, 2))
```

If you define the function in the test method, you may also use `self.variant`
as a decorator in the function definition. For example:

```python
class ExampleTest(chex.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test(self):
    @self.variant
    def var_fn(x, y):
       return x + y

    self.assertEqual(var_fn(1, 2), 3)
```

Example of parameterized test:

```python
from absl.testing import parameterized

# Could also be:
#  `class ExampleParameterizedTest(chex.TestCase, parameterized.TestCase):`
#  `class ExampleParameterizedTest(chex.TestCase):`
class ExampleParameterizedTest(parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ('case_positive', 1, 2, 3),
      ('case_negative', -1, -2, -3),
  )
  def test(self, arg_1, arg_2, expected):
    @self.variant
    def var_fn(x, y):
       return x + y

    self.assertEqual(var_fn(arg_1, arg_2), expected)
```

Chex currently supports the following variants:

* `with_jit` -- applies `jax.jit()` transformation to the function.
* `without_jit` -- uses the function as is, i.e. identity transformation.
* `with_device` -- places all arguments (except specified in `ignore_argnums`
   argument) into device memory before applying the function.
* `without_device` -- places all arguments in RAM before applying the function.
* `with_pmap` -- applies `jax.pmap()` transformation to the function (see notes below).

See documentation in [variants.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants.py) for more details on the supported variants.
More examples can be found in [variants_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants_test.py).

### Variants notes

* Test classes that use `@chex.variants` must inherit from
`chex.TestCase` (or any other base class that unrolls tests generators
within `TestCase`, e.g. `absl.testing.parameterized.TestCase`).

* **[`jax.vmap`]** All variants can be applied to a vmapped function;
please see an example in [variants_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants_test.py) (`test_vmapped_fn_named_params` and
`test_pmap_vmapped_fn`).

* **[`@chex.all_variants`]** You can get all supported variants
by using the decorator `@chex.all_variants`.

* **[`with_pmap` variant]** `jax.pmap(fn)`
([doc](https://jax.readthedocs.io/en/latest/jax.html#jax.pmap)) performs
parallel map of `fn` onto multiple devices. Since most tests run in a
single-device environment (i.e. having access to a single CPU or GPU), in which
case `jax.pmap` is a functional equivalent to `jax.jit`, ` with_pmap` variant is
skipped by default (although it works fine with a single device). Below we
describe  a way to properly test `fn` if it is supposed to be used in
multi-device environments (TPUs or multiple CPUs/GPUs). To disable skipping
`with_pmap` variants in case of a single device, add
`--chex_skip_pmap_variant_if_single_device=false` to your test command.

## Faking multi-device test environments

In situations where you do not have easy access to multiple devices, you can
still test parallel computation using single-device multi-threading.

In particular, one can force XLA to use a single CPU's threads as separate
devices, i.e. to fake a real multi-device environment with a multi-threaded one.
These two options are theoretically equivalent from XLA perspective because they
expose the same interface and use identical abstractions.

Chex has a flag `chex_n_cpu_devices` that specifies a number of CPU threads to
use as XLA devices.

To set up a multi-threaded XLA environment for `absl` tests, define
`setUpModule` function in your test module:

```python
def setUpModule():
  chex.set_n_cpu_devices()
```

Now you can launch your test with `python test.py --chex_n_cpu_devices=N` to run
it in multi-device regime. Note that **all** tests within a module will have an
access to `N` devices.

More examples can be found in [variants_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants_test.py), [fake_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake_test.py) and [fake_set_n_cpu_devices_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake_set_n_cpu_devices_test.py).

## Citing Chex

To cite this repository:

```
@software{chex2020github,
  author = {David Budden and Matteo Hessel and Iurii Kemaev and Stephen Spencer
            and Fabio Viola},
  title = {Chex: Testing made fun, in JAX!},
  url = {http://github.com/deepmind/chex},
  version = {0.0.1},
  year = {2020},
}
```

In this bibtex entry, the version number is intended to be from
[chex/\_\_init\_\_.py](https://github.com/deepmind/chex/blob/master/chex/__init__.py),
and the year corresponds to the project's open-source release.
