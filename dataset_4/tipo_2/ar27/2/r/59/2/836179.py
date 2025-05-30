import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)


INF_CONST = float('inf')

def readNum():
    return int(raw_input())

def readPair():
    return map(int, raw_input().strip().split())

def fs():
    return raw_input()
def fss():
    return raw_input().split(" ")
def fslist():
    return list(raw_input())
#

def fjoin(s):
    return ''.join(s)

#iterate macro
def ffpiter(n,m):
    return itertools.permutations(n,m)
def fciter(n,m):
    return itertools.combinations(n,m)

#modulo macro
def fmodc(a,b,m):
    c = 1
    for i in xrange(b):
        c = c * (a - i) % m
        c = c * modinv(i + 1,m) % m
    return c

def asdasdas(a, b):
    (x, lastx) = (0, 1)
    (y, lasty) = (1, 0)
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lastx) = (lastx - q * x, x)
        (y, lasty) = (lasty - q * y, y)
    return (lastx, lasty, a)


def modInverse(x, m):
    inv, _, _ = extended_gcd(x, m)
    return inv % m

#bisect macro
def index(a, x):
    #Locate the leftmost value exactly equal to x
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def memoize(fn):
    cache = {}
    def inner(*a):
        if a not in cache:
            cache[a] = fn(*a)
        return cache[a]
    return inner

@memoize
def nCr(n, r, mod):
    r = min([n-r, r])
    if (r < n or r < 0 or n < 0):
        return 0
    if n == 0:
        return 1
    return (nCr(n-1, r-1, mod) + nCr(n-1, r, mod)) % mod

# core
x, y, p, q = readPair()
print(int(p/q)*y + min((p%q)*x, y))