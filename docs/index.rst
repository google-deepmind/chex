:github_url: https://github.com/google-deepmind/chex/tree/main/docs

Chex
-----

Chex is a library of utilities for helping to write reliable JAX code.

This includes utils to help:

* Instrument your code (e.g. assertions)
* Debug (e.g. transforming pmaps in vmaps within a context manager).
* Test JAX code across many variants (e.g. jitted vs non-jitted).

Modules overview can be found `on GitHub <https://github.com/google-deepmind/chex#modules-overview>`_.

Installation
------------

Chex can be installed with pip directly from github, with the following command:

``pip install git+git://github.com/google-deepmind/chex.git``

or from PyPI:

``pip install chex``

.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api

Citing Chex
-----------

This repository is part of the `DeepMind JAX Ecosystem <https://deepmind.com/blog/article/using-jax-to-accelerate-our-research>`_.

To cite Chex please use the citation:

.. code-block:: bibtex

   @software{deepmind2020jax,
     title = {The {D}eep{M}ind {JAX} {E}cosystem},
     author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
     url = {http://github.com/deepmind},
     year = {2020},
   }

Contribute
----------

- `Issue tracker <https://github.com/google-deepmind/chex/issues>`_
- `Source code <https://github.com/google-deepmind/chex/tree/main>`_

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google-deepmind/chex/issues>`_.

License
-------

Chex is licensed under the Apache 2.0 License.


Indices and Tables
==================

* :ref:`genindex`
