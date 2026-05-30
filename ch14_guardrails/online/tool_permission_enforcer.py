"""
tool_permission_enforcer.py — Zero-trust tool execution with role-based permissions.

The confused deputy problem: your agent has API keys and database access that
attackers don't. If an attacker tricks the agent into calling a destructive tool,
the agent does the dirty work with its own credentials.

This enforcer wraps every tool call with:
  - Role-based permission checks (read / write / admin)
  - Argument validation before execution
  - Full audit log of every tool invocation
  - Hard blocks on high-risk argument patterns (path traversal, shell injection)

Usage:
    registry = ToolRegistry(role="analyst")
    registry.register("read_file", read_file_fn, permission="read")
    registry.register("write_file", write_file_fn, permission="write")

    result = registry.execute("read_file", {"path": "report.csv"})
    # Raises PermissionError if role doesn't allow this tool
    # Raises ValueError if arguments look dangerous
"""

import re
import json
import datetime
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum


class Permission(str, Enum):
    READ = "read"       # Safe, non-mutating operations
    WRITE = "write"     # Mutating operations (create, update, delete)
    ADMIN = "admin"     # Dangerous operations (execute code, delete records)


# Permission hierarchy: each role includes all lower-level permissions
ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    "viewer":  {Permission.READ},
    "analyst": {Permission.READ, Permission.WRITE},
    "admin":   {Permission.READ, Permission.WRITE, Permission.ADMIN},
}

# Argument patterns that should never reach a tool regardless of role
BLOCKED_ARG_PATTERNS = [
    r"\.\./",           # Path traversal
    r"/etc/(passwd|shadow|hosts)",
    r";\s*(rm|dd|mkfs|shred)",  # Shell injection
    r"\$\(",            # Command substitution
    r"`[^`]+`",         # Backtick execution
    r"--\s*drop\s+table",  # SQL injection
    r"xp_cmdshell",     # SQL Server shell execution
]


@dataclass
class AuditEntry:
    timestamp: str
    role: str
    tool: str
    args: dict
    permission: str
    allowed: bool
    blocked_reason: str | None = None
    result_summary: str | None = None


@dataclass
class ToolDefinition:
    name: str
    fn: Callable
    permission: Permission
    description: str = ""


class ToolRegistry:
    """
    Zero-trust tool registry. Every call is checked, logged, and validated.

    Never hand this registry directly to the LLM. The LLM should only see
    tool names and descriptions. Call registry.execute() from your agent
    loop after the LLM decides which tool to use.
    """

    def __init__(self, role: str = "viewer"):
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Unknown role: {role}. Valid: {list(ROLE_PERMISSIONS)}")
        self.role = role
        self._allowed_permissions = ROLE_PERMISSIONS[role]
        self._tools: dict[str, ToolDefinition] = {}
        self._audit_log: list[AuditEntry] = []

    def register(
        self,
        name: str,
        fn: Callable,
        permission: Permission = Permission.READ,
        description: str = "",
    ):
        self._tools[name] = ToolDefinition(
            name=name, fn=fn, permission=permission, description=description
        )

    def execute(self, tool_name: str, args: dict) -> Any:
        """Execute a tool with full permission and safety checks."""
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        # 1. Tool must be registered
        if tool_name not in self._tools:
            self._log(timestamp, tool_name, args, "none", allowed=False,
                      blocked_reason="Tool not found")
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]

        # 2. Role must have permission for this tool
        if tool.permission not in self._allowed_permissions:
            reason = (
                f"Role '{self.role}' has {[p.value for p in self._allowed_permissions]} "
                f"but '{tool_name}' requires '{tool.permission.value}'"
            )
            self._log(timestamp, tool_name, args, tool.permission.value,
                      allowed=False, blocked_reason=reason)
            raise PermissionError(reason)

        # 3. Argument safety check — block known attack patterns
        args_str = json.dumps(args)
        for pattern in BLOCKED_ARG_PATTERNS:
            if re.search(pattern, args_str, re.IGNORECASE):
                reason = f"Dangerous argument pattern detected: {pattern}"
                self._log(timestamp, tool_name, args, tool.permission.value,
                          allowed=False, blocked_reason=reason)
                raise ValueError(reason)

        # 4. Execute
        try:
            result = tool.fn(**args)
            result_summary = str(result)[:200] if result else "None"
            self._log(timestamp, tool_name, args, tool.permission.value,
                      allowed=True, result_summary=result_summary)
            return result
        except Exception as e:
            self._log(timestamp, tool_name, args, tool.permission.value,
                      allowed=True, blocked_reason=f"Execution error: {e}")
            raise

    def get_audit_log(self) -> list[AuditEntry]:
        return list(self._audit_log)

    def print_audit_log(self):
        print(f"\n{'='*60}")
        print(f"AUDIT LOG — role={self.role}, {len(self._audit_log)} entries")
        print(f"{'='*60}")
        for entry in self._audit_log:
            status = "✅ ALLOWED" if entry.allowed else "🚨 BLOCKED"
            print(f"[{entry.timestamp}] {status} {entry.tool}({entry.args})")
            if entry.blocked_reason:
                print(f"  Reason: {entry.blocked_reason}")

    def tool_catalog(self) -> list[dict]:
        """Return tool definitions safe to show to the LLM."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
            if t.permission in self._allowed_permissions
        ]

    def _log(self, timestamp, tool, args, permission, allowed, **kwargs):
        self._audit_log.append(AuditEntry(
            timestamp=timestamp, role=self.role, tool=tool,
            args=args, permission=permission, allowed=allowed, **kwargs
        ))


# --- Example usage ---
if __name__ == "__main__":
    # Simulated tool functions
    def read_file(path: str) -> str:
        return f"[contents of {path}]"

    def write_file(path: str, content: str) -> str:
        return f"[wrote {len(content)} bytes to {path}]"

    def delete_record(table: str, record_id: int) -> str:
        return f"[deleted {table}:{record_id}]"

    # Analyst can read and write, but not delete
    registry = ToolRegistry(role="analyst")
    registry.register("read_file", read_file, Permission.READ, "Read a file")
    registry.register("write_file", write_file, Permission.WRITE, "Write a file")
    registry.register("delete_record", delete_record, Permission.ADMIN, "Delete a DB record")

    print("Tools visible to analyst:", [t["name"] for t in registry.tool_catalog()])

    # Allowed: read
    result = registry.execute("read_file", {"path": "report.csv"})
    print(f"\nread_file result: {result}")

    # Allowed: write
    result = registry.execute("write_file", {"path": "output.txt", "content": "hello"})
    print(f"write_file result: {result}")

    # Blocked: insufficient role
    try:
        registry.execute("delete_record", {"table": "users", "record_id": 42})
    except PermissionError as e:
        print(f"\nPermissionError (expected): {e}")

    # Blocked: path traversal attack
    try:
        registry.execute("read_file", {"path": "../../etc/passwd"})
    except ValueError as e:
        print(f"ValueError (expected): {e}")

    registry.print_audit_log()
