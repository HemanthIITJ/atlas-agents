"""
sandbox_cost_tracker.py — Hard compute budgets for E2B sandbox executions.

Agents can infinite-loop. A while True inside a sandbox will happily burn
CPU until your monthly bill arrives. This module wraps an E2B sandbox with
per-execution timing and a cumulative budget cap.

When the budget is exceeded it raises BudgetExceededError and kills the
sandbox immediately — no clean-up delay, no grace period.

Run the demo:
    E2B_API_KEY=... python sandbox_cost_tracker.py

Requires: pip install e2b-code-interpreter
"""

import time
from dataclasses import dataclass, field
from e2b_code_interpreter import Sandbox


class BudgetExceededError(Exception):
    pass


@dataclass
class ExecutionRecord:
    code_preview: str
    duration_s: float
    success: bool


@dataclass
class BudgetedSandbox:
    """
    Wraps an E2B sandbox with a per-session compute budget.

    budget_s: maximum total CPU-seconds allowed across all executions.
    per_exec_timeout_s: hard timeout on a single execution (kills it if exceeded).
    """
    budget_s: float = 30.0
    per_exec_timeout_s: float = 10.0
    _spent_s: float = field(default=0.0, init=False)
    _sandbox: Sandbox = field(default=None, init=False)
    _history: list[ExecutionRecord] = field(default_factory=list, init=False)

    def __enter__(self):
        self._sandbox = Sandbox(timeout=int(self.per_exec_timeout_s) + 5)
        return self

    def __exit__(self, *_):
        if self._sandbox:
            self._sandbox.kill()

    @property
    def spent_s(self) -> float:
        return self._spent_s

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.budget_s - self._spent_s)

    def run(self, code: str) -> str:
        """Execute code, track time, raise BudgetExceededError if over budget."""
        if self._spent_s >= self.budget_s:
            raise BudgetExceededError(
                f"Budget exhausted: {self._spent_s:.1f}s used of {self.budget_s:.1f}s"
            )

        preview = code.strip()[:60].replace("\n", " ")
        start = time.perf_counter()

        try:
            result = self._sandbox.run_code(code)
            duration = time.perf_counter() - start
            self._spent_s += duration
            self._history.append(ExecutionRecord(preview, duration, success=not result.error))

            if self._spent_s >= self.budget_s:
                raise BudgetExceededError(
                    f"Budget exceeded after this execution: "
                    f"{self._spent_s:.1f}s of {self.budget_s:.1f}s used"
                )

            if result.error:
                return f"ERROR: {result.error}"
            return result.text or ""

        except BudgetExceededError:
            raise
        except Exception as e:
            duration = time.perf_counter() - start
            self._spent_s += duration
            self._history.append(ExecutionRecord(preview, duration, success=False))
            return f"EXCEPTION: {e}"

    def report(self):
        print(f"\n{'─'*50}")
        print(f"  Compute Budget Report")
        print(f"{'─'*50}")
        print(f"  Budget:    {self.budget_s:.1f}s")
        print(f"  Used:      {self._spent_s:.2f}s")
        print(f"  Remaining: {self.remaining_s:.2f}s")
        print(f"\n  Executions ({len(self._history)} total):")
        for i, rec in enumerate(self._history, 1):
            status = "✅" if rec.success else "❌"
            print(f"    {i}. {status} {rec.duration_s:.2f}s — {rec.code_preview}")
        print(f"{'─'*50}\n")


def main():
    print("Demo: 5-second budget, 3 code blocks\n")

    try:
        with BudgetedSandbox(budget_s=5.0, per_exec_timeout_s=4.0) as sb:

            print("Block 1: fast computation")
            out = sb.run("print(sum(range(1_000_000)))")
            print(f"  → {out.strip()}  ({sb.spent_s:.2f}s used)\n")

            print("Block 2: fast computation")
            out = sb.run("import math; print(math.factorial(20))")
            print(f"  → {out.strip()}  ({sb.spent_s:.2f}s used)\n")

            print("Block 3: deliberate sleep — will exceed the budget")
            out = sb.run("import time; time.sleep(10); print('done')")
            print(f"  → {out.strip()}  ({sb.spent_s:.2f}s used)\n")

            print("Block 4: this should never run")
            sb.run("print('should not reach here')")

    except BudgetExceededError as e:
        print(f"\n🛑 {e}")
        print("Sandbox killed. No further charges.")

    # The context manager already killed the sandbox — report is still available
    # In production you'd persist the history to your cost dashboard
    print("\nFinal report:")
    # Re-instantiate just for the report demo
    tracker = BudgetedSandbox(budget_s=5.0)
    tracker._spent_s = 5.3
    tracker._history = [
        ExecutionRecord("print(sum(range(1_000_000)))", 0.8, True),
        ExecutionRecord("import math; print(math.factorial(20))", 0.4, True),
        ExecutionRecord("import time; time.sleep(10); print('done')", 4.1, False),
    ]
    tracker.report()


if __name__ == "__main__":
    main()
