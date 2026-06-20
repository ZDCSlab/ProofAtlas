from __future__ import annotations

import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

from .utils import load_config, stable_hash, write_json


class PipelineTimer:
    def __init__(self, config_path: str | None = None) -> None:
        self._start = perf_counter()
        self.config_path = config_path
        self.config_hash = None
        if config_path:
            self.config_hash = stable_hash(json.dumps(load_config(config_path), sort_keys=True), 16)
        self.stages: list[dict[str, Any]] = []

    def run_stage(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        status = "passed"
        error = None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end = perf_counter()
            self.stages.append(
                {
                    "name": name,
                    "status": status,
                    "seconds": float(end - start),
                    "error": error,
                    "order": len(self.stages) + 1,
                }
            )

    def skip_stage(self, name: str, reason: str = "cached_artifacts_present") -> None:
        self.stages.append(
            {
                "name": name,
                "status": "skipped",
                "seconds": 0.0,
                "error": None,
                "order": len(self.stages) + 1,
                "reason": reason,
            }
        )

    def report(self) -> dict[str, Any]:
        total = perf_counter() - self._start
        executed_stage_count = sum(1 for stage in self.stages if stage.get("status") == "passed")
        skipped_stage_count = sum(1 for stage in self.stages if stage.get("status") == "skipped")
        return {
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_seconds": float(total),
            "stage_count": len(self.stages),
            "executed_stage_count": executed_stage_count,
            "skipped_stage_count": skipped_stage_count,
            "has_skipped_stages": skipped_stage_count > 0,
            "passed": all(stage.get("status") in {"passed", "skipped"} for stage in self.stages),
            "slowest_stages": sorted(
                [
                    {"name": stage["name"], "seconds": stage["seconds"]}
                    for stage in self.stages
                ],
                key=lambda row: row["seconds"],
                reverse=True,
            )[:10],
            "stages": self.stages,
        }

    def write(self, output_path: str = "outputs/reports/pipeline_run_timings.json") -> dict[str, Any]:
        report = self.report()
        write_json(output_path, report)
        return report
