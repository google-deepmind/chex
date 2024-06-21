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

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
rm -rf _testing
rm -rf .pytype
mkdir -p _testing
readonly VENV_DIR="$(mktemp -d -p `pwd`/_testing chex-env.XXXXXXXX)"
# in the unlikely case in which there was something in that directory
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pylint pylint-exit
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-test.txt

# Install the requested JAX version
if [ "$JAX_VERSION" = "" ]; then
  : # use version installed in requirements above
elif [ "$JAX_VERSION" = "newest" ]; then
  pip install -U jax jaxlib
else
  pip install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

# Lint with flake8.
flake8 `find chex -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Append specific config lines.
echo "signature-mutators=toolz.functoolz.curry" >> .pylintrc
echo "disable=unnecessary-lambda-assignment,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find chex -name '*.py' | grep -v 'test.py' | xargs` -d E1102|| pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find chex -name '*_test.py' | xargs` -d W0212,E1130,E1102 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
python setup.py sdist
pip wheel --verbose --no-deps --no-clean dist/chex*.tar.gz
pip install chex*.whl

# Check types with pytype.
# Note: pytype does not support 3.11 as of 25.06.23
# See https://github.com/google/pytype/issues/1308
if [ `python -c 'import sys; print(sys.version_info.minor)'` -lt 11 ];
then
  pip install pytype
  pytype `find chex/_src -name "*py" | xargs` -k
fi;

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
pip install -r requirements/requirements-test.txt
cd _testing

# Main tests.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs chex -k "not fake_set_n_cpu_devices_test"

# Isolate tests that use `chex.set_n_cpu_device()`.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs chex -k "fake_set_n_cpu_devices_test"
cd ..

# Build Sphinx docs.

pip install -r requirements/requirements-docs.txt
cd docs
make coverage_check
make html
cd ..

# cleanup
rm -rf _testing

set +u
deactivate
echo "All tests passed. Congrats!"
