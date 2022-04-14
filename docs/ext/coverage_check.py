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
"""Asserts all public symbols are covered in the docs."""

import inspect
import types
from typing import Any, Mapping, Sequence, Tuple

import chex as _module
from sphinx import application
from sphinx import builders
from sphinx import errors


def find_internal_python_modules(
    root_module: types.ModuleType,) -> Sequence[Tuple[str, types.ModuleType]]:
  """Returns `(name, module)` for all submodules under `root_module`."""
  modules = set([(root_module.__name__, root_module)])
  visited = set()
  to_visit = [root_module]

  while to_visit:
    mod = to_visit.pop()
    visited.add(mod)

    for name in dir(mod):
      obj = getattr(mod, name)
      if inspect.ismodule(obj) and obj not in visited:
        if obj.__name__.startswith(_module.__name__):
          if '_src' not in obj.__name__:
            to_visit.append(obj)
            modules.add((obj.__name__, obj))

  return sorted(modules)


def get_public_symbols() -> Sequence[Tuple[str, types.ModuleType]]:
  names = set()
  for module_name, module in find_internal_python_modules(_module):
    for name in module.__all__:
      names.add(module_name + '.' + name)
  return tuple(names)


class CoverageCheck(builders.Builder):
  """Builder that checks all public symbols are included."""

  name = 'coverage_check'

  def get_outdated_docs(self) -> str:
    return 'coverage_check'

  def write(self, *ignored: Any) -> None:
    pass

  def finish(self) -> None:
    documented_objects = frozenset(self.env.domaindata['py']['objects'])
    undocumented_objects = set(get_public_symbols()) - documented_objects

    # Exclude deprecated API symbols.
    assertion_exceptions = ('assert_tree_all_close',
                            'assert_tree_all_equal_comparator',
                            'assert_tree_all_equal_shapes',
                            'assert_tree_all_equal_structs')
    undocumented_objects -= {'chex.' + s for s in assertion_exceptions}

    # Exclude pytypes.
    pytypes_exceptions = ('Array', 'ArrayBatched', 'Array', 'ArrayBatched',
                          'ArrayDevice', 'ArrayDType', 'ArrayNumpy',
                          'ArraySharded', 'ArrayTree', 'Device', 'Numeric',
                          'PRNGKey', 'PyTreeDef', 'Scalar', 'Shape')
    undocumented_objects -= {'chex.' + s for s in pytypes_exceptions}

    if undocumented_objects:
      undocumented_objects = tuple(sorted(undocumented_objects))
      raise errors.SphinxError(
          'All public symbols must be included in our documentation, did you '
          'forget to add an entry to `api.rst`?\n'
          f'Undocumented symbols: {undocumented_objects}')


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.add_builder(CoverageCheck)
  return dict(version=_module.__version__, parallel_read_safe=True)
