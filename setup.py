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
"""Install script for setuptools."""

from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('chex/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=')+1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `chex/__init__.py`')


setup(
    name='chex',
    version=_get_version(),
    url='https://github.com/deepmind/chex',
    license='Apache 2.0',
    author='DeepMind',
    description=('Chex: Testing made fun, in JAX!'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='chex-dev@google.com',
    keywords='jax testing debugging python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=[
        'absl-py>=0.9.0',
        'dataclasses==0.7;python_version<"3.7"',
        'jax>=0.1.55',
        'jaxlib>=0.1.37',
        'numpy>=1.18.0',
        'toolz>=0.9.0',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Testing :: Mocking',
        'Topic :: Software Development :: Testing :: Unit',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
