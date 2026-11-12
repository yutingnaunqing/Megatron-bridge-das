# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Fail explicitly if pinned upstream methods differ from the validated base."""

import hashlib
from functools import lru_cache
from pathlib import Path


_EXPECTED = {
    "modelling_qwen3_vl/attention.py": "4838c8fc0d24a7915c7b52fc1341c3b3695f11084c5d53ce51b957c9cbdd9fd7",
    "modelling_qwen3_vl/transformer_block.py": "5733f8881963e12e816cbbd1676cbab854c6993ea87c58cc3e2977504482f013",
    "modelling_qwen3_vl/vision_model.py": "8fba8560f06ce9956655c4011777537ed0c706eb0535089679cadd9bf3d1e8a8",
    "modelling_qwen3_vl/transformer_config.py": "1a99783a18b900efd74ee187011f42d5b8a66676bcfef05d1d5eb02842fb12a9",
    "qwen35_vl_provider.py": "a9ee9c2dfd9351992e5c4a2fa50971cea113cac56d56c5ec5d3d3a3a8e340b33",
}


@lru_cache(maxsize=1)
def check_upstream_compatibility() -> None:
    """Verify the installed upstream implementation before binding HCU methods."""
    from megatron.bridge.models.qwen_vl import qwen35_vl_provider

    root = Path(qwen35_vl_provider.__file__).parent
    for name, expected in _EXPECTED.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unsupported upstream Qwen source: {name}; review HCU overrides before use")
