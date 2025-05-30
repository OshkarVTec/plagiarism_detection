# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import Literal as LitNode, Calculator as CalcEval, BinaryOp as BinOp, Visitor as BaseVisitor

# Tokens and literals
literals = ['+','-','*','/', '%', '(', ')', '{', '}', ';', '=']
reserved = {
    'while': 'WHILE',
    'if':    'IF',
    'else':  'ELSE',
    'int':   'INT',
    'bool':  'BOOL',
    'float': 'FLOAT',
    'char':  'CHAR',
    'main':  'MAIN',
}
tokens = [
    'INTLIT', 'ID', 'OR', 'AND', 'LEQ', 'GEQ', 'LT', 'GT',
    'EQ', 'NEQ', 'MINUS', 'PLUS', 'NOT',
] + list(reserved.values())

t_ignore = ' \t'

def t_identifier(tok):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    tok.type = reserved.get(tok.value, 'ID')
    return tok

def t_number(tok):
    r'[0-9]+'
    tok.value = int(tok.value)
    return tok

def t_linebreak(tok):
    r'\n+'
    tok.lexer.lineno += len(tok.value)

# Operators
OR    = r'\|\|'
AND   = r'&&'
LT    = r'<'
LEQ   = r'<='
GT    = r'>'
GEQ   = r'>='
EQ    = r'=='
NEQ   = r'!='
MINUS = r'-'
NOT   = r'!'
PLUS  = r'+'

def t_err(tok):
    print(f"Illegal character '{tok.value[0]}'")
    tok.lexer.skip(1)

#%%
# AST node definitions renamed
def VarNode(name):
    class _Var:
        def __init__(self, nm):
            self.nm = nm
        def accept(self, vst):
            vst.visit_variable(self)
    return _Var(name)

class UnaryOperator:
    def __init__(self, oper, expr):
        self.oper = oper
        self.expr = expr
    def accept(self, vst):
        vst.visit_unary(self)

class WhileLoop:
    def __init__(self, cond, stmt):
        self.cond = cond
        self.stmt = stmt
    def accept(self, vst):
        vst.visit_while(self)

class IfStmt:
    def __init__(self, cond, then_blk, else_blk=None):
        self.cond = cond
        self.then_blk = then_blk
        self.else_blk = else_blk
    def accept(self, vst):
        vst.visit_if(self)

class BlockNode:
    def __init__(self, stmts):
        self.stmts = stmts
    def accept(self, vst):
        vst.visit_block(self)

class AssignStmt:
    def __init__(self, idn, expr):
        self.idn = idn
        self.expr = expr
    def accept(self, vst):
        vst.visit_assign(self)

class DeclarationNode:
    def __init__(self, tpe, idn):
        self.tpe = tpe
        self.idn = idn
    def accept(self, vst):
        vst.visit_declare(self)

class ProgNode:
    def __init__(self, decls, stmts):
        self.decls = decls
        self.stmts = stmts
    def accept(self, vst):
        vst.visit_program(self)
# %%

precedence = (
    ('nonassoc', 'ELSE'),
    ('right', 'NOT', 'UMINUS'),
    ('left',  'OR'),
    ('left',  'AND'),
    ('nonassoc', 'EQ', 'NEQ'),
    ('nonassoc', 'LT', 'LEQ', 'GT', 'GEQ'),
    ('left',  'PLUS', 'MINUS'),
    ('left',  '*', '/', '%'),
)

def p_empty(p):
    'empty :'
    p[0] = []

def p_progDefinition(p):
    'program : INT MAIN "(" ")" "{" declarations stmts "}"'
    p[0] = ProgNode(p[6], p[7])

def p_vars(p):
    '''declarations : declaration declarations
                    | empty'''
    p[0] = ([p[1]] + p[2]) if len(p) == 3 else []

def p_declarationItem(p):
    'declaration : type ID ";"'
    p[0] = DeclarationNode(p[1], p[2])

def p_typeDef(p):
    '''type : INT
            | BOOL
            | FLOAT
            | CHAR'''
    p[0] = p.slice[1].type

def p_stmts(p):
    '''stmts : stmt stmts
             | empty'''
    p[0] = ([p[1]] + p[2]) if len(p) == 3 else []

def p_stmt(p):
    '''stmt : ';'
            | blockStmt
            | assignStmt
            | ifStmt
            | whileLoop'''
    p[0] = p[1]

def p_blockStmt(p):
    'blockStmt : "{" stmts "}"'
    p[0] = BlockNode(p[2])

def p_assignStmt(p):
    'assignStmt : ID "=" expr ";"'
    p[0] = AssignStmt(p[1], p[3])

#%%
source_code = """
int main() {
  int a;
  int b;
  bool fl;
  fl = -fl;
  fl = !fl ;
  b = 2 - 1;
  a = 0;
  if (a < 3) { a = a + 1; } else { a = a - 1; }
  while (a > 0) { a = a - 1; }
}
"""
lexerObj  = lex.lex()
parserObj = yacc.yacc(start='program')
syntax_tree = parserObj.parse(source_code)

# %%
from llvmlite import ir as irlib

class IRGen(BaseVisitor):
    def __init__(self):
        self.stack2 = []
        self.symbols2 = {}
        self.maps = {
            'INT':   irlib.IntType(32),
            'BOOL':  irlib.IntType(1),
            'FLOAT': irlib.DoubleType(),
            'CHAR':  irlib.IntType(8),
        }
    def visit_program(self, node):
        for d in node.decls:
            d.accept(self)
        for s in node.stmts:
            s.accept(self)
        builder2.ret(irlib.Constant(intType2, 0))


#%%
syntax_tree = parserObj.parse(source_code)
irgen = IRGen()
syntax_tree.accept(irgen)
print(ir_module2)
