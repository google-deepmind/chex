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
pip install --upgrade pip
pip install .[dev]

# Install the requested JAX version
if [ "$JAX_VERSION" = "" ]; then
  : # use version installed in requirements above
elif [ "$JAX_VERSION" = "newest" ]; then
  pip install -U jax jaxlib
elif [ "$JAX_VERSION" = "nightly" ]; then
  pip install -U --pre jax jaxlib --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/
else
  pip install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

# Lint with Ruff.
ruff check

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Enforce two space indent style.
sed -i "s/indent-string.*/indent-string='  '/" .pylintrc
# Append specific config lines.
echo "disable=unnecessary-lambda-assignment,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find chex -name '*.py' | grep -v 'test.py' | xargs` -d E1102|| pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find chex -name '*_test.py' | xargs` -d W0212,E1130,E1102,E1120 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
pip install build
python -m build
pip wheel --verbose --no-deps --no-clean dist/chex*.tar.gz
pip install chex*.whl --force-reinstall

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
pip install .[test]
cd _testing

# Main tests.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs chex -k "not fake_set_n_cpu_devices_test"

# Isolate tests that use `chex.set_n_cpu_device()`.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs chex -k "fake_set_n_cpu_devices_test"
cd ..

# Build Sphinx docs.
pip install .[docs]
cd docs
make coverage_check
make html
cd ..

# cleanup
rm -rf _testing

set +u
deactivate
echo "All tests passed. Congrats!"
