class MiniDBError(Exception):
    """Base exception for all MiniDB errors."""
    pass


class SchemaError(MiniDBError):
    """Raised when a table definition, type check, or constraint fails."""
    pass


class QuerySyntaxError(MiniDBError):
    """Raised when the custom query lexer or parser encounters invalid syntax."""
    pass


class ExecutionError(MiniDBError):
    """Raised when a query references missing tables, columns, or invalid operations."""
    pass


class TransactionError(MiniDBError):
    """Raised on invalid transaction states (e.g., nested transactions, commit without begin)."""
    pass


class CorruptionError(MiniDBError):
    """Raised when disk checksums fail during storage or WAL recovery."""
    pass