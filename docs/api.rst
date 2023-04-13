Assertions
==========

.. currentmodule:: chex

.. autosummary::

    assert_axis_dimension
    assert_axis_dimension_comparator
    assert_axis_dimension_gt
    assert_axis_dimension_gteq
    assert_axis_dimension_lt
    assert_axis_dimension_lteq
    assert_devices_available
    assert_equal
    assert_equal_rank
    assert_equal_shape
    assert_equal_shape_prefix
    assert_equal_shape_suffix
    assert_exactly_one_is_none
    assert_gpu_available
    assert_is_broadcastable
    assert_is_divisible
    assert_max_traces
    assert_not_both_none
    assert_numerical_grads
    assert_rank
    assert_scalar
    assert_scalar_in
    assert_scalar_negative
    assert_scalar_non_negative
    assert_scalar_positive
    assert_shape
    assert_tpu_available
    assert_tree_all_finite
    assert_tree_has_only_ndarrays
    assert_tree_is_on_device
    assert_tree_is_on_host
    assert_tree_is_sharded
    assert_tree_no_nones
    assert_tree_shape_prefix
    assert_tree_shape_suffix
    assert_trees_all_close
    assert_trees_all_close_ulp
    assert_trees_all_equal
    assert_trees_all_equal_comparator
    assert_trees_all_equal_dtypes
    assert_trees_all_equal_shapes
    assert_trees_all_equal_shapes_and_dtypes
    assert_trees_all_equal_structs
    assert_type
    chexify
    ChexifyChecks
    with_jittable_assertions
    block_until_chexify_assertions_complete
    Dimensions
    disable_asserts
    enable_asserts
    clear_trace_counter
    if_args_not_none


Jax Assertions
~~~~~~~~~~~~~~

.. autofunction:: assert_max_traces
.. autofunction:: assert_devices_available
.. autofunction:: assert_gpu_available
.. autofunction:: assert_tpu_available


Value (Runtime) Assertions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: chexify
.. autosummary::  ChexifyChecks
.. autofunction:: with_jittable_assertions
.. autofunction:: block_until_chexify_assertions_complete


Tree Assertions
~~~~~~~~~~~~~~~

.. autofunction:: assert_tree_all_finite
.. autofunction:: assert_tree_has_only_ndarrays
.. autofunction:: assert_tree_is_on_device
.. autofunction:: assert_tree_is_on_host
.. autofunction:: assert_tree_is_sharded
.. autofunction:: assert_tree_no_nones
.. autofunction:: assert_tree_shape_prefix
.. autofunction:: assert_tree_shape_suffix
.. autofunction:: assert_trees_all_close
.. autofunction:: assert_trees_all_close_ulp
.. autofunction:: assert_trees_all_equal
.. autofunction:: assert_trees_all_equal_comparator
.. autofunction:: assert_trees_all_equal_dtypes
.. autofunction:: assert_trees_all_equal_shapes
.. autofunction:: assert_trees_all_equal_shapes_and_dtypes
.. autofunction:: assert_trees_all_equal_structs


Generic Assertions
~~~~~~~~~~~~~~~~~~

.. autofunction:: assert_axis_dimension
.. autofunction:: assert_axis_dimension_comparator
.. autofunction:: assert_axis_dimension_gt
.. autofunction:: assert_axis_dimension_gteq
.. autofunction:: assert_axis_dimension_lt
.. autofunction:: assert_axis_dimension_lteq
.. autofunction:: assert_equal
.. autofunction:: assert_equal_rank
.. autofunction:: assert_equal_shape
.. autofunction:: assert_equal_shape_prefix
.. autofunction:: assert_equal_shape_suffix
.. autofunction:: assert_exactly_one_is_none
.. autofunction:: assert_is_broadcastable
.. autofunction:: assert_is_divisible
.. autofunction:: assert_not_both_none
.. autofunction:: assert_numerical_grads
.. autofunction:: assert_rank
.. autofunction:: assert_scalar
.. autofunction:: assert_scalar_in
.. autofunction:: assert_scalar_negative
.. autofunction:: assert_scalar_non_negative
.. autofunction:: assert_scalar_positive
.. autofunction:: assert_shape
.. autofunction:: assert_type


Shapes and Named Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Dimensions


Utils
~~~~~

.. autofunction:: disable_asserts
.. autofunction:: enable_asserts
.. autofunction:: clear_trace_counter
.. autofunction:: if_args_not_none


Backend restriction
===================

.. currentmodule:: chex

.. autofunction:: restrict_backends


Dataclasses
===========

.. currentmodule:: chex

.. autofunction:: dataclass
.. autofunction:: mappable_dataclass
.. autofunction:: register_dataclass_type_with_jax_tree_util


Fakes
=====

.. currentmodule:: chex

.. autosummary::

    fake_jit
    fake_pmap
    fake_pmap_and_jit
    set_n_cpu_devices

Transformations
~~~~~~~~~~~~~~~

.. autofunction:: fake_jit
.. autofunction:: fake_pmap
.. autofunction:: fake_pmap_and_jit


Devices
~~~~~~~

.. autofunction:: set_n_cpu_devices


Pytypes
=======

.. currentmodule:: chex

.. autosummary::

    Array
    ArrayBatched
    ArrayDevice
    ArrayDeviceTree
    ArrayDType
    ArrayNumpy
    ArrayNumpyTree
    ArraySharded
    ArrayTree
    Device
    Numeric
    PRNGKey
    PyTreeDef
    Scalar
    Shape



Variants
========
.. currentmodule:: chex

.. autoclass:: ChexVariantType
.. autoclass:: TestCase
.. autofunction:: variants
.. autofunction:: all_variants
.. autofunction:: params_product
