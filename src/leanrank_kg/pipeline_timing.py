from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from .utils import write_json


class PipelineTimer:
    def __init__(self) -> None:
        self._start = perf_counter()
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
        return {
            "total_seconds": float(total),
            "stage_count": len(self.stages),
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
