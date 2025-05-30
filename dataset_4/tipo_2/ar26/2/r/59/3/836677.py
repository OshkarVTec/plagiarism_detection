# Variante 1
import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)

MAX_VAL = float('inf')

def get_int():
    return int(raw_input())

def get_two():
    return map(int, raw_input().split())

def xs():
    return raw_input()
def xss():
    return raw_input().split(" ")
def xslist():
    return list(raw_input())

def jcoin(s):
    return ''.jcoin(s)

#iterate macro
def pditer(n,m):
    return itertools.permutations(n,m)
def cditer(n,m):
    return itertools.combinations(n,m)

#modulo macro
def modc(a,b,m):
    c = 1
    for i in xrange(b):
        c = c * (a - i) % m
        c = c * modinv(i + 1,m) % m
    return c
 

def ext_gcd(a, b):
    x0, x1 = 0, 1
    y0, y1 = 1, 0
    while b:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q*x0, x0
        y0, y1 = y1 - q*y0, y0
    return x1, y1, a

def inv(a, mod):
    i, j, g = ext_gcd(a, mod)
    return i % mod

def cache(fn):
    memo = {}
    def wrap(*args):
        if args not in memo:
            memo[args] = fn(*args)
        return memo[args]
    return wrap

# lógica principal
a, b, k, l = get_two()
print(int(k / l) * b + min((k % l) * a, b))
