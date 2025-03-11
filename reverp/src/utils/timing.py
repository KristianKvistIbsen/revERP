from time import perf_counter
from typing import Dict, Optional
from contextlib import contextmanager

class TimingTracker:
    """Tracks execution time of different analysis steps."""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._current_timer: Optional[float] = None
        self._current_step: Optional[str] = None

    @contextmanager
    def track(self, step_name: str):
        """Context manager for tracking time of a step."""
        try:
            self._current_timer = perf_counter()
            self._current_step = step_name
            yield
        finally:
            if self._current_timer is not None and self._current_step is not None:
                elapsed = perf_counter() - self._current_timer
                self.timings[self._current_step] = elapsed
                self._current_timer = None
                self._current_step = None

    def get_total_time(self) -> float:
        """Get total time across all tracked steps."""
        return sum(self.timings.values())

    def get_summary(self) -> str:
        """Generate timing summary string."""
        total = self.get_total_time()
        summary = ["Timing Summary:"]
        for step, time in self.timings.items():
            summary.append(f"{step}: {time:.2f}s ({(time/total)*100:.1f}%)")
        summary.append(f"Total Time: {total:.2f}s")
        return "\n".join(summary)