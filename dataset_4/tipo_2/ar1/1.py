# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import Literal, Calculator, BinaryOp, Visitor

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

t_ignore  = ' \t'

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_INTLIT(t):
    r'[0-9]+'
    t.value = int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

t_OR = r'\|\|'
t_AND = r'&&'
t_LT = r'<'
t_LEQ = r'<='
t_GT = r'>'
t_GEQ = r'>='
t_EQ = r'=='
t_NEQ = r'!='
t_MINUS = r'-'
t_NOT = r'!'
t_PLUS = r'\+'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

#%%
class Variable:
    def __init__(self, name):
        self.name = name

    def accept(self, visitor):
        visitor.visit_variable(self)

class UnaryOp:
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def accept(self, visitor):
        visitor.visit_unary_op(self)

class WhileStatement:
    def __init__(self, condition, statement):
        self.condition = condition
        self.statement = statement

    def accept(self, visitor):
        visitor.visit_while_statement(self)

class IfStatement:
    def __init__(self, condition, then_stmt, else_stmt=None):
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def accept(self, visitor):
        visitor.visit_if_statement(self)

class Block:
    def __init__(self, statements):
        self.statements = statements

    def accept(self, visitor):
        visitor.visit_block(self)

class Assignment:
    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression = expression

    def accept(self, visitor):
        visitor.visit_assignment(self)
        
class Declaration:
    def __init__(self, typ, identifier):
        self.typ = typ
        self.identifier = identifier
    def accept(self, visitor):
        visitor.visit_declaration(self)

class Program:
    def __init__(self, declarations, statements):
        self.declarations = declarations
        self.statements   = statements
    def accept(self, visitor):
        visitor.visit_program(self)
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
    '''empty :'''
    p[0] = []

def p_program(p):
    '''program : INT MAIN "(" ")" "{" declarations statements "}"'''
    p[0] = Program(p[6], p[7])

def p_declarations(p):
    '''declarations : declaration declarations
                    | empty'''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []

def p_declaration(p):
    '''declaration : type ID ";"'''
    p[0] = Declaration(p[1], p[2])

def p_type(p):
    '''type : INT
            | BOOL
            | FLOAT
            | CHAR'''
    p[0] = p.slice[1].type

def p_statements(p):
    '''statements : statement statements
                  | empty'''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []


def p_statement(p):
    '''statement : ';'
                 | block
                 | assignment
                 | if_statement
                 | while_statement'''
    p[0] = p[1]

def p_block(p):
    '''block : '{' statements '}' '''
    p[0] = Block(p[2])

def p_assignment(p):
    '''assignment : ID '=' expression ';' '''
    p[0] = Assignment(p[1], p[3])

# %%
data = """
int main() {
  int x;
  int y;
  bool flag;
  flag = -flag;
  flag = !flag ;
  y = 2 - 1; 
  x = 0;
  if (x < 3) { x = x + 1; } else { x = x - 1; }
  while (x > 0) { x = x - 1; }
}
"""
lexer  = lex.lex()
parser = yacc.yacc(start='program')
ast = parser.parse(data)

# %%
from llvmlite import ir

class IRGenerator(Visitor):
    def __init__(self):
        self.stack = []
        self.symbols = {}
        self.typemap = {
            'INT':   ir.IntType(32),
            'BOOL':  ir.IntType(1),
            'FLOAT': ir.DoubleType(),
            'CHAR':  ir.IntType(8),
        }

    def visit_program(self, node: Program):
        for decl in node.declarations:
            decl.accept(self)

        for stmt in node.statements:
            stmt.accept(self)

        builder.ret(ir.Constant(intType, 0))


#%%
ast = parser.parse(data)
visitor = IRGenerator()
ast.accept(visitor)
print(module)

