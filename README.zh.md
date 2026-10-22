# Chex (JAX 可靠性与测试实用工具库)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

![CI status](https://github.com/deepmind/chex/workflows/ci/badge.svg)
![docs](https://readthedocs.org/projects/chex/badge/?version=latest)
![pypi](https://img.shields.io/pypi/v/chex)

Chex 是一个用于编写高可靠 JAX 代码的实用工具库。

它包含以下功能的辅助工具：

*   **代码插桩与校验（Instrumentation）**：如断言（assertions）、告警（warnings）。
*   **调试（Debug）**：例如在上下文管理器中将 `pmap` 转换为 `vmap`。
*   **多变体测试（Test Variants）**：跨多种代码变体测试 JAX 代码（例如经 JIT 编译 vs 未经 JIT 编译）。

## 安装指南 (Installation)

您可以通过 PyPI 安装 Chex 的最新正式发布版本：

```sh
pip install chex
```

或者从 GitHub 安装最新的开发版本：

```sh
pip install git+https://github.com/deepmind/chex.git
```

## 模块概览 (Modules Overview)

### 数据类 Dataclass ([dataclass.py](https://github.com/deepmind/chex/blob/master/chex/_src/dataclass.py))

数据类（Dataclass）是 Python 3.7 引入的一种广泛使用的结构，能够以最少的样板代码轻松定义带类型的结构化数据。然而，原生数据类开箱并不能直接兼容 JAX 和 [dm-tree](https://github.com/deepmind/tree)。

在 Chex 中，我们通过复用 Python 原生 [dataclasses](https://docs.python.org/3/library/dataclasses.html#module-dataclasses)，提供了对 JAX 友好的数据类实现。

Chex 的 `dataclass` 实现将数据类注册为内部 [_PyTree_ 节点](https://jax.readthedocs.io/en/latest/pytrees.html)，以确保与 JAX 数据结构的完全兼容。

此外，我们提供了一个类包装器，将数据类暴露为 `collections.Mapping` 的子类，从而允许使用 `dm-tree` 方法像处理常规 Python 字典一样处理它们（例如展平 flatten 或解展平 unflatten）。更多细节请参阅 [`@mappable_dataclass`](https://github.com/deepmind/chex/blob/master/chex/_src/dataclass.py#L27) 的文档字符串。

示例：

```python
@chex.dataclass
class Parameters:
  x: chex.ArrayDevice
  y: chex.ArrayDevice

parameters = Parameters(
    x=jnp.ones((2, 2)),
    y=jnp.ones((1, 2)),
)

# 数据类可作为 JAX pytrees 处理
jax.tree_util.tree_map(lambda x: 2.0 * x, parameters)

# 亦可作为 dm-tree 的 mapping 映射进行处理
tree.flatten(parameters)
```

**注意**：与标准 Python 3.7 数据类不同，Chex 数据类**不能**使用位置参数进行构造。它们支持与 Python `dict` 构造函数格式相同的传参方式。如有需要，数据类可以通过 `from_tuple` 和 `to_tuple` 方法与元组相互转换。

```python
parameters = Parameters(
    jnp.ones((2, 2)),
    jnp.ones((1, 2)),
)
# ValueError: Mappable dataclass constructor doesn't support positional args.
```

### 断言机制 Assertions ([asserts.py](https://github.com/deepmind/chex/blob/master/chex/_src/asserts.py))

JAX 的 PyType 类型注解的一个局限性在于，它们不支持指定 `DeviceArray` 的阶数（ranks）、形状（shapes）或数据类型（dtypes）。Chex 包含许多函数，能够灵活且简洁地对这些属性进行规范与校验。

例如，假设您希望确保所有张量 `t1`、`t2`、`t3` 具有相同的形状，并且张量 `t4`、`t5` 的阶数分别为 `2` 和（`3` 或 `4`）：

```python
chex.assert_equal_shape([t1, t2, t3])
chex.assert_rank([t4, t5], [2, {3, 4}])
```

更多示例：

```python
from chex import assert_shape, assert_rank, ...

assert_shape(x, (2, 3))                # x 的形状为 (2, 3)
assert_shape([x, y], [(), (2,3)])      # x 为标量，y 的形状为 (2, 3)

assert_rank(x, 0)                      # x 为标量（阶数为 0）
assert_rank([x, y], [0, 2])            # x 为标量，y 为 2 阶数组
assert_rank([x, y], {0, 2})            # x 和 y 均为标量或 2 阶数组

assert_type(x, int)                    # x 的类型为 `int`（x 可以是数组）
assert_type([x, y], [int, float])      # x 为 `int` 类型，y 为 `float` 类型

assert_equal_shape([x, y, z])          # x、y 和 z 具有相同的形状

assert_trees_all_close(tree_x, tree_y) # 树的值与结构完全匹配
assert_tree_all_finite(tree_x)         # tree_x 的所有叶子节点均为有限值（非 NaN/Inf）

assert_devices_available(2, 'gpu')     # 存在 2 个可用 GPU
assert_tpu_available()                 # 至少存在 1 个可用 TPU

assert_numerical_grads(f, (x, y), j)   # f^{(j)}(x, y) 与数值梯度一致
```

查阅 `asserts.py` [官方文档](https://chex.readthedocs.io/en/latest/api.html#assertions) 获取所有支持的断言函数列表。

如果您找不到特定的断言，欢迎提交 Pull Request 或在[问题追踪器](https://github.com/deepmind/chex/issues)上提出 Issue。

#### 可选参数 (Optional Arguments)

所有 Chex 断言均支持以下用于控制抛出异常消息的可选关键字参数：

*   `custom_message`：包含在抛出异常消息中的自定义字符串。
*   `include_default_message`：是否在抛出的异常消息中包含 Chex 的默认提示信息。
*   `exception_type`：要使用的异常类型，默认为 `AssertionError`。

例如以下代码：

```python
dataset = load_dataset()
params = init_params()
for i in range(num_steps):
  params = update_params(params, dataset.sample())
  chex.assert_tree_all_finite(params,
                              custom_message=f'Failed at iteration {i}.',
                              exception_type=ValueError)
```

当 `params` 被 `NaN` 或 `None` 污染时，将抛出包含当前迭代步数的 `ValueError`。

#### 静态断言与值断言（运行时断言） (Static and Value Assertions)

Chex 将所有断言划分为两类：**静态（static）**断言与**值（value）**断言。

1.  **静态（static）**断言：仅依据张量的元数据（除具体数值以外的信息）进行判断。例如：`assert_shape`、`assert_trees_all_equal_dtypes`、`assert_max_traces`。
2.  **值（value）**断言：需要访问张量的具体数值，而这些数值在 JAX 追踪（tracing）期间是不可用的（参阅 [JAX 原语工作原理](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html)），因此在 **JIT 编译**代码中需要特殊处理。

若要在经 JIT 编译的函数中启用值断言，可以使用 `chex.chexify()` 包装器进行装饰。示例：

```python
  @chex.chexify
  @jax.jit
  def logp1_abs_safe(x: chex.Array) -> chex.Array:
    chex.assert_tree_all_finite(x)
    return jnp.log(jnp.abs(x) + 1)

  logp1_abs_safe(jnp.ones(2))  # 正常运行
  logp1_abs_safe(jnp.array([jnp.nan, 3]))  # 抛出异常（在异步模式下）

  # 错误将在下一行代码或下一次调用 `logp1_abs_safe` 时抛出。
  # 有关异步模式的详细信息，请参阅官方文档。
  logp1_abs_safe.wait_checks()  # 等待（异步）计算完成。
```

有关 `chex.chexify()` 的更多详情，请参阅[此文档字符串](https://chex.readthedocs.io/en/latest/api.html#chex.chexify)。

#### JAX 追踪断言 (JAX Tracing Assertions)

每当传递给 JIT 编译函数的参数结构发生变化时，JAX 都会对该函数重新进行追踪（re-trace）。通常这种行为是无意识发生的，会导致严重的性能下降，且往往极难排查。[@chex.assert_max_traces](https://github.com/deepmind/chex/blob/master/chex/_src/asserts.py#L44) 装饰器可以断言该函数在程序执行期间重新追踪的次数不超过 `n` 次。

全局追踪计数器可以通过调用 `chex.clear_trace_counter()` 清除。该函数可用于隔离依赖 `@chex.assert_max_traces` 的单元测试。

示例：

```python
  @jax.jit
  @chex.assert_max_traces(n=1)
  def fn_sum_jitted(x, y):
    return x + y

  fn_sum_jitted(jnp.zeros(3), jnp.zeros(3))  # 首次追踪 - 正常运行
  fn_sum_jitted(jnp.zeros([6, 7]), jnp.zeros([6, 7]))  # 触发 AssertionError！
```

亦可与 `jax.pmap()` 配合使用：

```python
  def fn_sub(x, y):
    return x - y

  fn_sub_pmapped = jax.pmap(chex.assert_max_traces(fn_sub, n=10))
```

有关追踪的更多信息，请参阅 [JAX 原语工作原理](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html) 章节。

### 告警机制 Warnings ([warnigns.py](https://github.com/deepmind/chex/blob/master/chex/_src/warnings.py))

除了硬性断言之外，Chex 还提供了添加常见告警的实用工具，例如特定类型的弃用警告（deprecation warnings）。

### 测试变体 Test variants ([variants.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants.py))

JAX 高度依赖代码变换与即时编译，这意味着很难保证代码得到充分的测试。例如，仅仅测试一个调用了 JAX 代码的 Python 函数，并不会覆盖 JIT 编译时实际执行的代码路径；并且，该路径在针对 CPU、GPU 或 TPU 编译时也各不相同。这曾是许多隐蔽且难以捕捉的 bug 的根源——XLA 的改动可能导致非预期的异常行为，但这些行为仅在某一种特定的代码变换下才会暴露。

变体（Variants）提供了一个简洁的装饰器，可以在所有（或选定的）相关代码变换下重复运行任何测试，从而确保单元测试覆盖函数的不同“变体”。

例如，假设您希望测试函数 `fn` 在有无 JIT 编译下的输出。您可以使用 `chex.variants` 分别在函数的 JIT 编译版和非 JIT 编译版下运行测试，只需使用 `@chex.variants` 装饰测试方法，并在测试体中使用 `self.variant(fn)` 代替 `fn`：

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

如果在测试方法内部定义函数，您也可以直接在函数定义处将 `self.variant` 作为装饰器使用。例如：

```python
class ExampleTest(chex.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test(self):
    @self.variant
    def var_fn(x, y):
       return x + y

    self.assertEqual(var_fn(1, 2), 3)
```

参数化测试示例：

```python
from absl.testing import parameterized

# 亦可写作：
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

Chex 目前支持以下变体：

*   `with_jit` —— 对函数应用 `jax.jit()` 变换。
*   `without_jit` —— 原样使用函数，即恒等变换。
*   `with_device` —— 在应用函数之前，将所有参数（`ignore_argnums` 中指定的除外）放入设备内存中。
*   `without_device` —— 在应用函数之前，将所有参数保留在系统内存（RAM）中。
*   `with_pmap` —— 对函数应用 `jax.pmap()` 变换（详见下方说明）。

有关支持变体的更多详细信息，请参阅 [variants.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants.py) 中的文档。更多示例可在 [variants_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants_test.py) 中找到。

### 变体使用说明 (Variants notes)

*   使用 `@chex.variants` 的测试类必须继承自 `chex.TestCase`（或任何在 `TestCase` 内部展开测试生成器的基类，例如 `absl.testing.parameterized.TestCase`）。
*   **[`jax.vmap`]**：所有变体均可应用于经过 `vmap` 变换的函数；请参阅 [variants_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants_test.py) 中的示例（`test_vmapped_fn_named_params` 和 `test_pmap_vmapped_fn`）。
*   **[`@chex.all_variants`]**：可以通过使用 `@chex.all_variants` 装饰器直接获取所有受支持的变体。
*   **[`with_pmap` 变体]**：`jax.pmap(fn)`（[文档](https://jax.readthedocs.io/en/latest/jax.html#jax.pmap)）将 `fn` 并行映射到多个设备上。由于大多数测试都在单设备环境（即只能访问单个 CPU 或 GPU）中运行，此时 `jax.pmap` 在功能上等效于 `jax.jit`，因此默认情况下跳过 `with_pmap` 变体（尽管它在单设备下亦可正常工作）。下文介绍了在需要在多设备环境（TPU 或多个 CPU/GPU）中使用的场景下正确测试 `fn` 的方法。若要在单设备情况下禁用跳过 `with_pmap` 变体，请在测试命令中添加 `--chex_skip_pmap_variant_if_single_device=false`。

### 模拟测试 Fakes ([fake.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake.py))

JAX 中的调试由于 `jit` 和 `pmap` 等代码变换而变得更加困难，这些变换引入的优化使得代码难以审查和追踪。在调试过程中禁用这些变换通常也很繁琐，因为它们可能在底层代码的多处被调用。Chex 提供了工具，可以全局将 `jax.jit` 替换为无操作（no-op）变换，将 `jax.pmap` 替换为（非并行的）`jax.vmap`，以便在单设备上下文中更轻松地调试代码。

例如，您可以使用 Chex 模拟 `pmap` 并将其替换为 `vmap`。可以通过使用上下文管理器包裹代码来实现：

```python
with chex.fake_pmap():
  @jax.pmap
  def fn(inputs):
    ...

  # 函数将在 inputs 上被 vmap 向量化映射
  fn(inputs)
```

同样的功能也可以通过 `start` 和 `stop` 方法调用：

```python
fake_pmap = chex.fake_pmap()
fake_pmap.start()
... 您的 jax 代码 ...
fake_pmap.stop()
```

此外，您还可以使用多线程 CPU 模拟真实的多设备测试环境。详见**模拟多设备测试环境**章节。

更多详细信息请参阅 [fake.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake.py) 中的文档以及 [fake_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake_test.py) 中的示例。

## 模拟多设备测试环境 (Faking multi-device test environments)

在无法轻易访问多个设备的场景下，您仍可以使用单设备多线程测试并行计算。

具体而言，可以强制 XLA 将单个 CPU 的线程视为独立的设备，即通过多线程环境模拟真实的多设备环境。这两种方式在 XLA 的视角下在理论上是等效的，因为它们暴露了相同的接口并使用了完全一致的抽象。

Chex 提供了标志 `--chex_n_cpu_devices`，用于指定作为 XLA 设备使用的 CPU 线程数。

若要为 `absl` 测试设置多线程 XLA 环境，请在测试模块中定义 `setUpModule` 函数：

```python
def setUpModule():
  chex.set_n_cpu_devices()
```

现在，您可以使用 `python test.py --chex_n_cpu_devices=N` 启动测试，以在多设备模式下运行。请注意，模块内的**所有**测试都将拥有访问 `N` 个设备的权限。

更多示例请参阅 [variants_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/variants_test.py)、[fake_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake_test.py) 以及 [fake_set_n_cpu_devices_test.py](https://github.com/deepmind/chex/blob/master/chex/_src/fake_set_n_cpu_devices_test.py)。

### 检测错误后端的使用 ([restrict_backends.py](https://github.com/deepmind/chex/blob/master/chex/_src/restrict_backends.py))

有时，某些 JAX 代码需要在存在加速器（如 GPU/TPU）但加速器被预留给其他任务的环境中运行。通常开发者会使用 `jax.jit(..., backend='cpu')` 来使代码避开加速器，但很难手动确保整个子系统中毫无例外地贯彻了这一点。此时，可以使用 `restrict_backends()` 来检测任何被遗漏的情况，并通过抛出异常进行报告。

`restrict_backends()` 是一个上下文管理器，它会拒绝针对指定后端的 JAX 编译。例如：

```python
  with chex.restrict_backends(allowed=['cpu']):
    call_some_jax_subsystem_that_must_only_run_on_cpu()

  with chex.restrict_backends(forbidden=['tpu']):
    call_some_other_subsystem_that_must_not_touch_any_tpu()
```

### 使用具名维度大小 (Using named dimension sizes)

Chex 提供了一个小工具，允许将一组维度大小打包为一个对象。基本用法如下：

```python
dims = chex.Dimensions(B=batch_size, T=sequence_len, E=embedding_dim)
...
chex.assert_shape(arr, dims['BTE'])
```

字符串键查询会被转换为整数元组。例如，假设 `batch_size == 3`、`sequence_len = 5` 且 `embedding_dim = 7`，则：

```python
dims['BTE'] == (3, 5, 7)
dims['B'] == (3,)
dims['TTBEE'] == (5, 5, 3, 7, 7)
...
```

您还可以按如下方式动态分配维度大小：

```python
dims['XY'] = some_matrix.shape
dims.Z = 13
```

更多示例请参阅 [chex.Dimensions](https://chex.readthedocs.io/en/latest/api.html#chex.Dimensions) 官方文档。

## 引用 Chex (Citing Chex)

本仓库是 [DeepMind JAX 生态系统](http://github.com/deepmind) 的一部分。如需引用 Chex，请使用以下 BibTeX 条目：

```bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/deepmind},
  year = {2020},
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月22日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
