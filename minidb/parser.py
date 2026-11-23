import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from .errors import QuerySyntaxError

TOKEN_SPEC = [
    ("FLOAT", r"-?\d+\.\d+"),
    ("INT", r"-?\d+"),
    ("STRING", r"'([^']*)'"),
    ("NAMED_PARAM", r":[A-Za-z_][A-Za-z0-9_]*"),
    ("QMARK", r"\?"),
    ("OP", r"!=|<=|>=|=|<|>"),
    ("COMMA", r","),
    ("DOT", r"\."),
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
    "CREATE", "DROP", "ALTER", "ADD", "RENAME", "COLUMN", "TO", "DEFAULT",
    "REFERENCES", "CASCADE", "RESTRICT",
    "SHOW", "TABLES", "DESCRIBE",
    "TABLE", "INDEX", "ON", "USING",
    "INSERT", "INTO", "VALUES",
    "SELECT", "DISTINCT", "AS", "FROM", "JOIN", "INNER", "LEFT", "WHERE", "AND", "OR", "IN", "LIKE",
    "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "LIMIT",
    "UPDATE", "SET", "DELETE",
    "EXPLAIN", "VACUUM",
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
        elif kind == "NAMED_PARAM":
            tokens.append(Token("NAMED_PARAM", val[1:], pos))
        elif kind == "QMARK":
            tokens.append(Token("QMARK", "?", pos))
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
    # (col_name, data_type, not_null, unique, foreign_key_tuple)
    columns: List[Tuple[str, str, bool, bool, Optional[Tuple[str, str, str]]]]


@dataclass
class AlterTableStmt:
    table_name: str
    action: str  # "ADD_COLUMN", "DROP_COLUMN", "RENAME_COLUMN"
    column_def: Optional[Tuple[str, str, bool, bool, Optional[Tuple[str, str, str]]]] = None
    default_value: Any = None
    has_default: bool = False
    column_name: Optional[str] = None
    new_column_name: Optional[str] = None


@dataclass
class DropTableStmt:
    table_name: str


@dataclass
class CreateIndexStmt:
    index_name: str
    table_name: str
    columns: List[str]
    index_type: str


@dataclass
class DropIndexStmt:
    index_name: str


@dataclass
class ShowTablesStmt:
    pass


@dataclass
class DescribeTableStmt:
    table_name: str


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
    connectors: List[str]


@dataclass
class HavingCondition:
    target_type: str  # "AGG" or "COLUMN"
    field: str
    op: str
    value: Any
    func: Optional[str] = None


@dataclass
class HavingClause:
    conditions: List[HavingCondition]
    connectors: List[str]


@dataclass
class JoinClause:
    join_type: str  # "INNER" or "LEFT"
    right_table: str
    left_col: str
    right_col: str


@dataclass
class SelectItem:
    expr_type: str
    field: Optional[str]
    func: Optional[str] = None
    alias: Optional[str] = None


@dataclass
class SelectStmt:
    items: List[SelectItem]
    table_name: str
    joins: List[JoinClause]
    where: Optional[WhereClause]
    group_by: Optional[str]
    having: Optional[HavingClause]
    order_by: Optional[Tuple[str, bool]]
    limit: Optional[int]
    distinct: bool = False


@dataclass
class UpdateStmt:
    table_name: str
    assignments: List[Tuple[str, Any]]
    where: Optional[WhereClause]


@dataclass
class DeleteStmt:
    table_name: str
    where: Optional[WhereClause]


@dataclass
class ExplainStmt:
    select_stmt: SelectStmt


@dataclass
class VacuumStmt:
    table_name: str


class Parser:
    def __init__(self, tokens: List[Token], params: Any = None):
        self.tokens = tokens
        self.pos = 0
        self.params = params
        self._param_idx = 0

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

    def _parse_col_ref(self) -> str:
        first = self.expect_type("IDENT").value
        if self.current().type == "DOT":
            self.advance()
            second = self.expect_type("IDENT").value
            return f"{first}.{second}"
        return first

    def _parse_references_clause(self) -> Tuple[str, str, str]:
        ref_table = self.expect_type("IDENT").value
        self.expect_type("LPAREN")
        ref_col = self.expect_type("IDENT").value
        self.expect_type("RPAREN")
        on_delete = "RESTRICT"
        if self.match_keyword("ON"):
            self.expect_keyword("DELETE")
            if self.match_keyword("CASCADE"):
                on_delete = "CASCADE"
            elif self.match_keyword("RESTRICT"):
                on_delete = "RESTRICT"
            else:
                raise QuerySyntaxError("Expected CASCADE or RESTRICT after ON DELETE.")
        return (ref_table, ref_col, on_delete)

    def parse(self) -> Any:
        tok = self.current()
        if tok.type != "KEYWORD":
            raise QuerySyntaxError(f"Queries must start with a command keyword, got {tok.value!r}.")

        cmd = tok.value
        if cmd == "CREATE":
            stmt = self._parse_create()
        elif cmd == "ALTER":
            stmt = self._parse_alter()
        elif cmd == "DROP":
            stmt = self._parse_drop()
        elif cmd == "SHOW":
            self.advance()
            self.expect_keyword("TABLES")
            stmt = ShowTablesStmt()
        elif cmd == "DESCRIBE":
            self.advance()
            stmt = DescribeTableStmt(table_name=self.expect_type("IDENT").value)
        elif cmd == "INSERT":
            stmt = self._parse_insert()
        elif cmd == "SELECT":
            stmt = self._parse_select()
        elif cmd == "UPDATE":
            stmt = self._parse_update()
        elif cmd == "DELETE":
            stmt = self._parse_delete()
        elif cmd == "EXPLAIN":
            self.advance()
            stmt = ExplainStmt(select_stmt=self._parse_select())
        elif cmd == "VACUUM":
            self.advance()
            stmt = VacuumStmt(table_name=self.expect_type("IDENT").value)
        else:
            raise QuerySyntaxError(f"Unsupported statement starting with '{cmd}'.")

        if self.current().type == "SEMI":
            self.advance()

        if self.current().type != "EOF":
            raise QuerySyntaxError(f"Unexpected trailing token {self.current().value!r} at pos {self.current().pos}.")

        if isinstance(self.params, (list, tuple)) and self._param_idx != len(self.params):
            raise QuerySyntaxError(
                f"Too many parameters provided: expected {self._param_idx}, got {len(self.params)}."
            )

        return stmt

    def _parse_literal(self) -> Any:
        tok = self.current()
        if tok.type == "QMARK":
            self.advance()
            if not isinstance(self.params, (list, tuple)):
                raise QuerySyntaxError("Positional '?' parameter requires a tuple or list of parameters.")
            if self._param_idx >= len(self.params):
                raise QuerySyntaxError(
                    f"Missing value for positional parameter #{self._param_idx + 1} at pos {tok.pos}."
                )
            val = self.params[self._param_idx]
            self._param_idx += 1
            return val

        if tok.type == "NAMED_PARAM":
            self.advance()
            if not isinstance(self.params, dict):
                raise QuerySyntaxError(f"Named parameter ':{tok.value}' requires a dict of parameters.")
            if tok.value not in self.params:
                raise QuerySyntaxError(f"Missing parameter ':{tok.value}' in parameter dict.")
            return self.params[tok.value]

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

    def _parse_alter(self) -> AlterTableStmt:
        self.expect_keyword("ALTER")
        self.expect_keyword("TABLE")
        table_name = self.expect_type("IDENT").value

        if self.match_keyword("ADD"):
            self.match_keyword("COLUMN")
            col_name = self.expect_type("IDENT").value
            type_tok = self.expect_type("KEYWORD")
            if type_tok.value not in ("INT", "FLOAT", "STR", "BOOL"):
                raise QuerySyntaxError(f"Invalid column type '{type_tok.value}'.")
            not_null = False
            unique = False
            has_default = False
            default_value = None
            foreign_key = None
            while self.current().type == "KEYWORD" and self.current().value in ("NOT", "UNIQUE", "DEFAULT", "REFERENCES"):
                if self.match_keyword("NOT"):
                    self.expect_keyword("NULL")
                    not_null = True
                elif self.match_keyword("UNIQUE"):
                    unique = True
                elif self.match_keyword("DEFAULT"):
                    has_default = True
                    default_value = self._parse_literal()
                elif self.match_keyword("REFERENCES"):
                    foreign_key = self._parse_references_clause()
            return AlterTableStmt(
                table_name=table_name,
                action="ADD_COLUMN",
                column_def=(col_name, type_tok.value, not_null, unique, foreign_key),
                default_value=default_value,
                has_default=has_default,
            )

        if self.match_keyword("DROP"):
            self.match_keyword("COLUMN")
            col_name = self.expect_type("IDENT").value
            return AlterTableStmt(
                table_name=table_name,
                action="DROP_COLUMN",
                column_name=col_name,
            )

        if self.match_keyword("RENAME"):
            self.match_keyword("COLUMN")
            old_col = self.expect_type("IDENT").value
            self.expect_keyword("TO")
            new_col = self.expect_type("IDENT").value
            return AlterTableStmt(
                table_name=table_name,
                action="RENAME_COLUMN",
                column_name=old_col,
                new_column_name=new_col,
            )

        raise QuerySyntaxError("Expected ADD, DROP, or RENAME after ALTER TABLE.")

    def _parse_drop(self) -> Any:
        self.expect_keyword("DROP")
        if self.match_keyword("TABLE"):
            return DropTableStmt(table_name=self.expect_type("IDENT").value)
        elif self.match_keyword("INDEX"):
            return DropIndexStmt(index_name=self.expect_type("IDENT").value)
        raise QuerySyntaxError("Expected TABLE or INDEX after DROP.")

    def _parse_create(self) -> Any:
        self.expect_keyword("CREATE")
        if self.match_keyword("TABLE"):
            table_name = self.expect_type("IDENT").value
            self.expect_type("LPAREN")
            cols: List[Tuple[str, str, bool, bool, Optional[Tuple[str, str, str]]]] = []
            while True:
                col_name = self.expect_type("IDENT").value
                type_tok = self.expect_type("KEYWORD")
                if type_tok.value not in ("INT", "FLOAT", "STR", "BOOL"):
                    raise QuerySyntaxError(f"Invalid column type '{type_tok.value}'.")
                not_null = False
                unique = False
                foreign_key = None
                while self.current().type == "KEYWORD" and self.current().value in ("NOT", "UNIQUE", "REFERENCES"):
                    if self.match_keyword("NOT"):
                        self.expect_keyword("NULL")
                        not_null = True
                    elif self.match_keyword("UNIQUE"):
                        unique = True
                    elif self.match_keyword("REFERENCES"):
                        foreign_key = self._parse_references_clause()
                cols.append((col_name, type_tok.value, not_null, unique, foreign_key))
                if self.current().type == "COMMA":
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
            idx_cols: List[str] = []
            while True:
                idx_cols.append(self.expect_type("IDENT").value)
                if self.current().type == "COMMA":
                    self.advance()
                else:
                    break
            self.expect_type("RPAREN")
            index_type = "BTREE"
            if self.match_keyword("USING"):
                idx_tok = self.expect_type("IDENT")
                index_type = idx_tok.value.upper()
                if index_type not in ("BTREE", "HASH"):
                    raise QuerySyntaxError(f"Index type must be BTREE or HASH, got {index_type!r}.")
            return CreateIndexStmt(index_name=index_name, table_name=table_name, columns=idx_cols, index_type=index_type)

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
            field = self._parse_col_ref()
            if self.match_keyword("IN"):
                self.expect_type("LPAREN")
                in_vals: List[Any] = []
                while True:
                    in_vals.append(self._parse_literal())
                    if self.current().type == "COMMA":
                        self.advance()
                    else:
                        break
                self.expect_type("RPAREN")
                conditions.append(Condition(field=field, op="IN", value=in_vals))
            elif self.match_keyword("LIKE"):
                pattern = self._parse_literal()
                if not isinstance(pattern, str):
                    raise QuerySyntaxError("LIKE operator requires a string pattern.")
                conditions.append(Condition(field=field, op="LIKE", value=pattern))
            else:
                op_tok = self.expect_type("OP")
                val = self._parse_literal()
                conditions.append(Condition(field=field, op=op_tok.value, value=val))

            if self.current().type == "KEYWORD" and self.current().value in ("AND", "OR"):
                connectors.append(self.advance().value)
            else:
                break

        return WhereClause(conditions=conditions, connectors=connectors)

    def _parse_having(self) -> Optional[HavingClause]:
        if not self.match_keyword("HAVING"):
            return None
        conditions: List[HavingCondition] = []
        connectors: List[str] = []

        while True:
            if self.current().type == "KEYWORD" and self.current().value in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                func = self.advance().value
                self.expect_type("LPAREN")
                if self.current().type == "STAR":
                    self.advance()
                    field = "*"
                else:
                    field = self._parse_col_ref()
                self.expect_type("RPAREN")
                op_tok = self.expect_type("OP")
                val = self._parse_literal()
                conditions.append(
                    HavingCondition(target_type="AGG", field=field, op=op_tok.value, value=val, func=func)
                )
            else:
                field = self._parse_col_ref()
                op_tok = self.expect_type("OP")
                val = self._parse_literal()
                conditions.append(
                    HavingCondition(target_type="COLUMN", field=field, op=op_tok.value, value=val)
                )

            if self.current().type == "KEYWORD" and self.current().value in ("AND", "OR"):
                connectors.append(self.advance().value)
            else:
                break

        return HavingClause(conditions=conditions, connectors=connectors)

    def _parse_select(self) -> SelectStmt:
        self.expect_keyword("SELECT")
        distinct = self.match_keyword("DISTINCT")
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
                    field = self._parse_col_ref()
                self.expect_type("RPAREN")
                alias = f"{func}({field})"
                if self.match_keyword("AS"):
                    alias = self.expect_type("IDENT").value
                items.append(SelectItem(expr_type="AGG", field=field, func=func, alias=alias))
            else:
                field = self._parse_col_ref()
                alias = field
                if self.match_keyword("AS"):
                    alias = self.expect_type("IDENT").value
                items.append(SelectItem(expr_type="COLUMN", field=field, alias=alias))

            if self.current().type == "COMMA":
                self.advance()
            else:
                break

        self.expect_keyword("FROM")
        table_name = self.expect_type("IDENT").value

        joins: List[JoinClause] = []
        while self.current().type == "KEYWORD" and self.current().value in ("JOIN", "INNER", "LEFT"):
            join_type = "INNER"
            if self.match_keyword("LEFT"):
                join_type = "LEFT"
            elif self.match_keyword("INNER"):
                join_type = "INNER"
            self.expect_keyword("JOIN")
            right_table = self.expect_type("IDENT").value
            self.expect_keyword("ON")
            left_col = self._parse_col_ref()
            op = self.expect_type("OP")
            if op.value != "=":
                raise QuerySyntaxError(f"JOIN ON clause only supports '=' equality, got {op.value!r}.")
            right_col = self._parse_col_ref()
            joins.append(JoinClause(join_type=join_type, right_table=right_table, left_col=left_col, right_col=right_col))

        where = self._parse_where()

        group_by: Optional[str] = None
        if self.match_keyword("GROUP"):
            self.expect_keyword("BY")
            group_by = self._parse_col_ref()

        having = self._parse_having()
        if having is not None and group_by is None and not any(i.expr_type == "AGG" for i in items):
            raise QuerySyntaxError("HAVING clause requires GROUP BY or aggregate functions in SELECT.")

        order_by: Optional[Tuple[str, bool]] = None
        if self.match_keyword("ORDER"):
            self.expect_keyword("BY")
            if self.current().type == "KEYWORD" and self.current().value in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                func = self.advance().value
                self.expect_type("LPAREN")
                if self.current().type == "STAR":
                    self.advance()
                    inner = "*"
                else:
                    inner = self._parse_col_ref()
                self.expect_type("RPAREN")
                sort_field = f"{func}({inner})"
            else:
                sort_field = self._parse_col_ref()

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
            joins=joins,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            distinct=distinct,
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


def parse_query(query: str, params: Any = None) -> Any:
    tokens = tokenize(query)
    return Parser(tokens, params=params).parse()