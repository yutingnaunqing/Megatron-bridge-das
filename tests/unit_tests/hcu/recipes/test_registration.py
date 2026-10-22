# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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

"""Tests for how HCU recipes reach the upstream recipe runner.

``env/env.sh`` copies
``hcu_megatron/bridge/recipes/__init__.py`` over
``megatron/bridge/recipes/__init__.py`` before every launch, and the runner
then resolves ``--recipe`` with ``getattr`` on that module.  Two things must
therefore hold: the HCU file has to star-import every family it publishes, and
it has to keep star-importing every upstream family -- dropping one there would
quietly remove upstream recipes from the runner.

The upstream half reads the pristine file out of the submodule's git objects
rather than the working tree, because after a single launch the working-tree
file *is* the HCU file and comparing against it would pass no matter what.  It
is textual on purpose: this property breaks without any HCU code running, so it
must not depend on importing Megatron.
"""

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
HCU_RECIPES_INIT = REPO_ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py"
UPSTREAM_REPO = REPO_ROOT / "3rdparty/Megatron-Bridge"
UPSTREAM_RECIPES_INIT = "src/megatron/bridge/recipes/__init__.py"

# Every HCU family the runner must be able to resolve by name.
HCU_FAMILIES = ("deepseek", "qwen35_vl", "qwen38", "qwen38_vl")
EXPECTED_FAMILY_PREFIXES = ("deepseek_v4_", "qwen35_vl_", "qwen38_", "qwen38_vl_")

_STAR_IMPORT = re.compile(r"^from ([\w.]+) import \*$", re.MULTILINE)


def _star_imports(source: str) -> set:
    return set(_STAR_IMPORT.findall(source))


def _pristine_upstream_init() -> str:
    """The upstream recipe index as committed, unaffected by env.sh's copy."""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={UPSTREAM_REPO}", "-C", str(UPSTREAM_REPO),
         "show", f"HEAD:{UPSTREAM_RECIPES_INIT}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot read the pristine upstream recipe index: {result.stderr.strip()}")
    return result.stdout


def test_hcu_init_star_imports_every_family():
    imports = _star_imports(HCU_RECIPES_INIT.read_text(encoding="utf-8"))

    for family in HCU_FAMILIES:
        assert f"hcu_megatron.bridge.recipes.{family}" in imports, family


@pytest.mark.skipif(not UPSTREAM_REPO.exists(), reason="3rdparty/Megatron-Bridge is not checked out")
def test_hcu_init_keeps_every_upstream_import():
    """env.sh overwrites the upstream file, so anything dropped here is lost."""
    upstream = _star_imports(_pristine_upstream_init())
    hcu = _star_imports(HCU_RECIPES_INIT.read_text(encoding="utf-8"))

    assert upstream, "read no imports from the upstream recipe index"
    assert not upstream - hcu, f"HCU recipes __init__ would drop upstream families: {sorted(upstream - hcu)}"


def test_published_names_follow_the_runner_conventions():
    """Names must carry the family, mode, and suffix the runner parses them for."""
    import importlib

    for family in HCU_FAMILIES:
        module = importlib.import_module(f"hcu_megatron.bridge.recipes.{family}")
        assert module.__all__, family
        for recipe_name in module.__all__:
            assert recipe_name.startswith(EXPECTED_FAMILY_PREFIXES), recipe_name
            assert any(f"_{mode}_" in recipe_name for mode in ("pretrain", "sft")), recipe_name
            assert recipe_name.endswith("_config"), recipe_name
