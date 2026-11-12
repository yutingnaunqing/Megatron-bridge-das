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

"""HCU recipe registration without copying files into the upstream package."""

import importlib
import logging


_FAMILIES = ("deepseek", "glm", "minimax", "qwen35", "qwen35_vl", "qwen38", "qwen38_vl")


def register_hcu_recipes() -> tuple[str, ...]:
    """Expose the same HCU recipe exports to the existing library recipe lookup."""
    target = importlib.import_module("megatron.bridge.recipes")
    registered = []
    for family in _FAMILIES:
        module = importlib.import_module(f"hcu_megatron.bridge.recipes.{family}")
        for name in getattr(module, "__all__", dir(module)):
            value = getattr(module, name)
            if (
                name.startswith("_")
                or not callable(value)
                or not getattr(value, "__module__", "").startswith("hcu_megatron.")
            ):
                continue
            previous = getattr(target, name, None)
            if previous is not None and previous is not value:
                raise RuntimeError(f"HCU recipe name conflicts with existing registration: {name}")
            setattr(target, name, value)
            registered.append(name)
    logging.getLogger(__name__).info("Registered %d HCU recipes in process", len(registered))
    return tuple(registered)
