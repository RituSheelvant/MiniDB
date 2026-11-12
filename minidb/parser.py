import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from .errors import QuerySyntaxError

# Token specification (order matters)
TOKEN_SPEC = [
    ("FLOAT", r"-?\d+\.\d+"),
    ("INT", r"-?\d+"),
    ("STRING", r"'([^']*)'"),
    ("OP", r"!=|<=|>=|=|<|>"),
    ("COMMA", r","),
    ("SEMI", r";"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("STAR", r"\*"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("SKIP", r"[ \t\r\n]+"),
    ("MISMATCH", r"."),
]

MASTER_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))

KEYWORDS = {
    "CREATE", "TABLE", "INDEX", "ON", "USING",
    "INSERT", "INTO", "VALUES",
    "SELECT", "FROM", "WHERE", "AND", "OR",
    "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT",
    "UPDATE", "SET", "DELETE",
    "INT", "FLOAT", "STR", "BOOL",
    "NOT", "NULL", "UNIQUE", "TRUE", "FALSE",
    "COUNT", "SUM", "AVG", "MIN", "MAX",
}


@dataclass
class Token:
    type: str
    value: Any
    pos: int


def tokenize(query: str) -> List[Token]:
    tokens: List[Token] = []
    for match in MASTER_REGEX.finditer(query):
        kind = match.lastgroup
        val = match.group()
        pos = match.start()

        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise QuerySyntaxError(f"Unexpected character {val!r} at position {pos}.")
        elif kind == "INT":
            tokens.append(Token("INT", int(val), pos))
        elif kind == "FLOAT":
            tokens.append(Token("FLOAT", float(val), pos))
        elif kind == "STRING":
            tokens.append(Token("STRING", val[1:-1], pos))
        elif kind == "IDENT":
            upper = val.upper()
            if upper in KEYWORDS:
                tokens.append(Token("KEYWORD", upper, pos))
            else:
                tokens.append(Token("IDENT", val, pos))
        else:
            tokens.append(Token(kind, val, pos))

    tokens.append(Token("EOF", None, len(query)))
    return tokens


# --- AST Node Definitions ---

@dataclass
class CreateTableStmt:
    table_name: str
    columns: List[Tuple[str, str, bool, bool]]  # (name, type, not_null, unique)


@dataclass
class CreateIndexStmt:
    index_name: str
    table_name: str
    column_name: str
    index_type: str  # "BTREE" or "HASH"


@dataclass
class InsertStmt:
    table_name: str
    columns: Optional[List[str]]
    values: List[Any]


@dataclass
class Condition:
    field: str
    op: str
    value: Any


@dataclass
class WhereClause:
    conditions: List[Condition]
    connectors: List[str]  # "AND" / "OR" between conditions


@dataclass
class SelectItem:
    expr_type: str  # "STAR", "COLUMN", "AGG"
    field: Optional[str]
    func: Optional[str] = None  # COUNT, SUM, AVG, MIN, MAX
    alias: Optional[str] = None


@dataclass
class SelectStmt:
    items: List[SelectItem]
    table_name: str
    where: Optional[WhereClause]
    group_by: Optional[str]
    order_by: Optional[Tuple[str, bool]]  # (field_or_alias, is_desc)
    limit: Optional[int]


@dataclass
class UpdateStmt:
    table_name: str
    assignments: List[Tuple[str, Any]]
    where: Optional[WhereClause]


@dataclass
class DeleteStmt:
    table_name: str
    where: Optional[WhereClause]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def match_keyword(self, kw: str) -> bool:
        if self.current().type == "KEYWORD" and self.current().value == kw:
            self.advance()
            return True
        return False

    def expect_keyword(self, kw: str) -> Token:
        if self.current().type == "KEYWORD" and self.current().value == kw:
            return self.advance()
        raise QuerySyntaxError(f"Expected keyword '{kw}', got {self.current().value!r} at pos {self.current().pos}.")

    def expect_type(self, tok_type: str) -> Token:
        if self.current().type == tok_type:
            return self.advance()
        raise QuerySyntaxError(f"Expected {tok_type}, got {self.current().type} ({self.current().value!r}) at pos {self.current().pos}.")

    def parse(self) -> Any:
        tok = self.current()
        if tok.type != "KEYWORD":
            raise QuerySyntaxError(f"Queries must start with a command keyword, got {tok.value!r}.")

        cmd = tok.value
        if cmd == "CREATE":
            stmt = self._parse_create()
        elif cmd == "INSERT":
            stmt = self._parse_insert()
        elif cmd == "SELECT":
            stmt = self._parse_select()
        elif cmd == "UPDATE":
            stmt = self._parse_update()
        elif cmd == "DELETE":
            stmt = self._parse_delete()
        else:
            raise QuerySyntaxError(f"Unsupported statement starting with '{cmd}'.")

        if self.current().type == "SEMI":
            self.advance()

        if self.current().type != "EOF":
            raise QuerySyntaxError(f"Unexpected trailing token {self.current().value!r} at pos {self.current().pos}.")
        return stmt

    def _parse_literal(self) -> Any:
        tok = self.current()
        if tok.type in ("INT", "FLOAT", "STRING"):
            self.advance()
            return tok.value
        if tok.type == "KEYWORD" and tok.value == "TRUE":
            self.advance()
            return True
        if tok.type == "KEYWORD" and tok.value == "FALSE":
            self.advance()
            return False
        if tok.type == "KEYWORD" and tok.value == "NULL":
            self.advance()
            return None
        raise QuerySyntaxError(f"Expected literal value, got {tok.value!r} at pos {tok.pos}.")

    def _parse_create(self) -> Any:
        self.expect_keyword("CREATE")
        if self.match_keyword("TABLE"):
            table_name = self.expect_type("IDENT").value
            self.expect_type("LPAREN")
            cols: List[Tuple[str, str, bool, bool]] = []
            while True:
                col_name = self.expect_type("IDENT").value
                type_tok = self.expect_type("KEYWORD")
                if type_tok.value not in ("INT", "FLOAT", "STR", "BOOL"):
                    raise QuerySyntaxError(f"Invalid column type '{type_tok.value}'.")
                not_null = False
                unique = False
                while self.current().type == "KEYWORD" and self.current().value in ("NOT", "UNIQUE"):
                    if self.match_keyword("NOT"):
                        self.expect_keyword("NULL")
                        not_null = True
                    elif self.match_keyword("UNIQUE"):
                        unique = True
                cols.append((col_name, type_tok.value, not_null, unique))
                if not self.match_keyword(",") and self.current().type == "COMMA":
                    self.advance()
                else:
                    break
            self.expect_type("RPAREN")
            return CreateTableStmt(table_name=table_name, columns=cols)

        elif self.match_keyword("INDEX"):
            index_name = self.expect_type("IDENT").value
            self.expect_keyword("ON")
            table_name = self.expect_type("IDENT").value
            self.expect_type("LPAREN")
            col_name = self.expect_type("IDENT").value
            self.expect_type("RPAREN")
            index_type = "BTREE"
            if self.match_keyword("USING"):
                idx_tok = self.expect_type("IDENT")
                index_type = idx_tok.value.upper()
                if index_type not in ("BTREE", "HASH"):
                    raise QuerySyntaxError(f"Index type must be BTREE or HASH, got {index_type!r}.")
            return CreateIndexStmt(index_name=index_name, table_name=table_name, column_name=col_name, index_type=index_type)

        raise QuerySyntaxError("Expected TABLE or INDEX after CREATE.")

    def _parse_insert(self) -> InsertStmt:
        self.expect_keyword("INSERT")
        self.expect_keyword("INTO")
        table_name = self.expect_type("IDENT").value

        columns: Optional[List[str]] = None
        if self.current().type == "LPAREN":
            self.advance()
            columns = []
            while True:
                columns.append(self.expect_type("IDENT").value)
                if self.current().type == "COMMA":
                    self.advance()
                else:
                    break
            self.expect_type("RPAREN")

        self.expect_keyword("VALUES")
        self.expect_type("LPAREN")
        values: List[Any] = []
        while True:
            values.append(self._parse_literal())
            if self.current().type == "COMMA":
                self.advance()
            else:
                break
        self.expect_type("RPAREN")
        return InsertStmt(table_name=table_name, columns=columns, values=values)

    def _parse_where(self) -> Optional[WhereClause]:
        if not self.match_keyword("WHERE"):
            return None
        conditions: List[Condition] = []
        connectors: List[str] = []

        while True:
            field = self.expect_type("IDENT").value
            op_tok = self.expect_type("OP")
            val = self._parse_literal()
            conditions.append(Condition(field=field, op=op_tok.value, value=val))

            if self.current().type == "KEYWORD" and self.current().value in ("AND", "OR"):
                connectors.append(self.advance().value)
            else:
                break

        return WhereClause(conditions=conditions, connectors=connectors)

    def _parse_select(self) -> SelectStmt:
        self.expect_keyword("SELECT")
        items: List[SelectItem] = []

        while True:
            if self.current().type == "STAR":
                self.advance()
                items.append(SelectItem(expr_type="STAR", field="*"))
            elif self.current().type == "KEYWORD" and self.current().value in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                func = self.advance().value
                self.expect_type("LPAREN")
                if self.current().type == "STAR":
                    self.advance()
                    field = "*"
                else:
                    field = self.expect_type("IDENT").value
                self.expect_type("RPAREN")
                alias = f"{func}({field})"
                items.append(SelectItem(expr_type="AGG", field=field, func=func, alias=alias))
            else:
                field = self.expect_type("IDENT").value
                items.append(SelectItem(expr_type="COLUMN", field=field, alias=field))

            if self.current().type == "COMMA":
                self.advance()
            else:
                break

        self.expect_keyword("FROM")
        table_name = self.expect_type("IDENT").value
        where = self._parse_where()

        group_by: Optional[str] = None
        if self.match_keyword("GROUP"):
            self.expect_keyword("BY")
            group_by = self.expect_type("IDENT").value

        order_by: Optional[Tuple[str, bool]] = None
        if self.match_keyword("ORDER"):
            self.expect_keyword("BY")
            if self.current().type == "KEYWORD" and self.current().value in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                func = self.advance().value
                self.expect_type("LPAREN")
                inner = "*" if self.current().type == "STAR" else self.expect_type("IDENT").value
                if self.current().type == "STAR":
                    self.advance()
                self.expect_type("RPAREN")
                sort_field = f"{func}({inner})"
            else:
                sort_field = self.expect_type("IDENT").value

            is_desc = False
            if self.match_keyword("DESC"):
                is_desc = True
            elif self.match_keyword("ASC"):
                is_desc = False
            order_by = (sort_field, is_desc)

        limit: Optional[int] = None
        if self.match_keyword("LIMIT"):
            limit = self.expect_type("INT").value

        return SelectStmt(
            items=items,
            table_name=table_name,
            where=where,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
        )

    def _parse_update(self) -> UpdateStmt:
        self.expect_keyword("UPDATE")
        table_name = self.expect_type("IDENT").value
        self.expect_keyword("SET")
        assignments: List[Tuple[str, Any]] = []
        while True:
            col = self.expect_type("IDENT").value
            op = self.expect_type("OP")
            if op.value != "=":
                raise QuerySyntaxError(f"Expected '=' in SET clause, got {op.value!r}.")
            val = self._parse_literal()
            assignments.append((col, val))
            if self.current().type == "COMMA":
                self.advance()
            else:
                break
        where = self._parse_where()
        return UpdateStmt(table_name=table_name, assignments=assignments, where=where)

    def _parse_delete(self) -> DeleteStmt:
        self.expect_keyword("DELETE")
        self.expect_keyword("FROM")
        table_name = self.expect_type("IDENT").value
        where = self._parse_where()
        return DeleteStmt(table_name=table_name, where=where)


def parse_query(query: str) -> Any:
    tokens = tokenize(query)
    return Parser(tokens).parse()