"""
e2b_multi_language.py — Run Python, Go, and Bash in the same E2B sandbox.

The key insight: E2B is a full Linux microVM, not a Python REPL. The agent
can write code in any language, compile it, and run it — all inside the
same isolated environment. This file demonstrates three patterns:

1. Python via the native run_code() interpreter
2. Go via file write + compile + execute (the correct way, not heredoc)
3. Bash for system-level tasks (disk usage, process listing, etc.)

Run it:
    E2B_API_KEY=... python e2b_multi_language.py

Requires: pip install e2b-code-interpreter
"""

from e2b_code_interpreter import Sandbox


PYTHON_TASK = """
import json, statistics

sales = [12400, 15300, 11200, 18700, 14500, 21000, 19800, 22400, 17600, 20100, 23500, 26000]
monthly = {f"Month {i+1}": v for i, v in enumerate(sales)}

print(json.dumps({
    "total": sum(sales),
    "average": round(statistics.mean(sales), 2),
    "peak_month": f"Month {sales.index(max(sales)) + 1}",
    "peak_value": max(sales),
    "growth_pct": round((sales[-1] - sales[0]) / sales[0] * 100, 1),
}, indent=2))
"""

GO_TASK = """package main

import (
    "fmt"
    "math"
)

func isPrime(n int) bool {
    if n < 2 {
        return false
    }
    for i := 2; i <= int(math.Sqrt(float64(n))); i++ {
        if n%i == 0 {
            return false
        }
    }
    return true
}

func main() {
    primes := []int{}
    for i := 2; len(primes) < 10; i++ {
        if isPrime(i) {
            primes = append(primes, i)
        }
    }
    fmt.Println("First 10 primes:", primes)
}
"""

BASH_TASK = """
echo "=== System Info ==="
uname -a
echo ""
echo "=== Disk Usage ==="
df -h /
echo ""
echo "=== Memory ==="
free -h
"""


def run_python(sandbox: Sandbox) -> str:
    print("  Running Python...")
    result = sandbox.run_code(PYTHON_TASK)
    if result.error:
        return f"ERROR: {result.error}"
    return result.text.strip()


def run_go(sandbox: Sandbox) -> str:
    """Write Go source to a file, then compile and run via shell command."""
    print("  Running Go (write → compile → execute)...")

    # Write the source file — use files.write(), not heredoc in a shell string
    sandbox.files.write("main.go", GO_TASK.encode())

    result = sandbox.commands.run("go run main.go")
    if result.exit_code != 0:
        return f"ERROR (exit {result.exit_code}): {result.stderr}"
    return result.stdout.strip()


def run_bash(sandbox: Sandbox) -> str:
    print("  Running Bash...")
    result = sandbox.commands.run(BASH_TASK)
    if result.exit_code != 0:
        return f"ERROR: {result.stderr}"
    return result.stdout.strip()


def main():
    print("Spinning up E2B sandbox...")
    with Sandbox() as sandbox:
        print("Sandbox ready.\n")

        print("[Python]")
        print(run_python(sandbox))
        print()

        print("[Go]")
        print(run_go(sandbox))
        print()

        print("[Bash]")
        print(run_bash(sandbox))
        print()

    print("Sandbox terminated.")


if __name__ == "__main__":
    main()
