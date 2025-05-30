import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)

CONST_INF = float('inf')

def read_one():
    return int(raw_input())

def read_two():
    return map(int, raw_input().split())

def ns():
    return raw_input()
def nss():
    return raw_input().split(" ")
def snlist():
    return list(raw_input())
#

def jnoin(s):
    return ''.join(s)

#iterate macro
def pniter(n,m):
    return itertools.permutations(n,m)
def cniter(n,m):
    return itertools.combinations(n,m)

#modulo macro
def modcn(a,b,m):
    c = 1
    for i in xrange(b):
        c = c * (a - i) % m
        c = c * modinv(i + 1,m) % m
    return c

def gcssd(a, b):
    (x, lastx) = (0, 1)
    (y, lasty) = (1, 0)
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lastx) = (lastx - q * x, x)
        (y, lasty) = (lasty - q * y, y)
    return (lastx, lasty, a)

def inv_mod(a, mod):
    i, j, g = gcd_ext(a, mod)
    return i % mod


def find_pos(seq, val):
    #Locate the leftmost value exactly equal to x
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def memoize(f):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return wrapper

@memoize
def choose(n, k, m):
    k = min(k, n-k)
    if n < 0 or k < 0 or k > n: return 0
    if n == 0: return 1
    return (choose(n-1, k-1, m) + choose(n-1, k, m)) % m

# main logic
A, B, K, L = read_two()
print(int(K/L)*B + min((K%L)*A, B))
