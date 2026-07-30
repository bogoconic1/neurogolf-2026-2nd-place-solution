# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal ARC-GEN task registry for the NeuroGolf Hugging Face leaderboard."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path


MAPPING_PATH = Path(__file__).resolve().parent.parent / "mapping.csv"


def _mapped_arc_gen_ids() -> list[str]:
    with MAPPING_PATH.open(newline="") as handle:
        return sorted({row["arc_gen_task_id"] for row in csv.DictReader(handle)})


def task_list() -> dict[str, tuple]:
    tasks = {}
    for task_id in _mapped_arc_gen_ids():
        module = importlib.import_module(f"tasks.task_{task_id}")
        tasks[task_id] = (module.generate, module.validate)
    return tasks
