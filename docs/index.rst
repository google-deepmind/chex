:github_url: https://github.com/deepmind/chex/tree/master/docs

Chex
-----

Chex is a library of utilities for helping to write reliable JAX code.

This includes utils to help:

* Instrument your code (e.g. assertions)
* Debug (e.g. transforming pmaps in vmaps within a context manager).
* Test JAX code across many variants (e.g. jitted vs non-jitted).

Modules overview can be found `on GitHub <https://github.com/deepmind/chex#modules-overview>`_.

Installation
------------

Chex can be installed with pip directly from github, with the following command:

``pip install git+git://github.com/deepmind/chex.git``

or from PyPI:

``pip install chex``

.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api

Citing Chex
-----------

This repository is part of the `DeepMind JAX Ecosystem <https://deepmind.com/blog/article/using-jax-to-accelerate-our-research>`_.

To cite Chex please use the `DeepMind JAX Ecosystem citation <https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt>`_.

Contribute
----------

- `Issue tracker <https://github.com/deepmind/chex/issues>`_
- `Source code <https://github.com/deepmind/chex/tree/master>`_

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/chex/issues>`_.

License
-------

Chex is licensed under the Apache 2.0 License.


Indices and Tables
==================

* :ref:`genindex`
