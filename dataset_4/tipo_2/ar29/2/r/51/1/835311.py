import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)

INF = float('inf')

def getInt(): 
    return int(raw_input())
def getTwo():
    return map(int, raw_input().split())
def ms():
    return raw_input()
def mss():
    return raw_input().split(" ")
def mslist():
    return list(raw_input())
#

def mjoin(s):
    return ''.join(s)

#iterate macro
def mpiter(n,m):
    return itertools.permutations(n,m)
def mciter(n,m):
    return itertools.combinations(n,m)

#modulo macro
def mmodc(a,b,m):
    c = 1
    for i in xrange(b):
        c = c * (a - i) % m
        c = c * modinv(i + 1,m) % m
    return c

def invMod(a, m):
    inv, _, _ = egcd(a, m)
    return inv % m


def bin_index(arr, val):
    pos = bisect.bisect_left(arr, val)
    if pos < len(arr) and arr[pos] == val:
        return pos
    return -1


def cache_it(f):
    mem = {}
    def f2(*args):
        if args not in mem:
            mem[args] = f(*args)
        return mem[args]
    return f2

@cache_it
def nC(n, k, m):
    k = min(k, n-k)
    if k<0 or k>n: return 0
    if n==0: return 1
    return (nC(n-1, k-1, m)+nC(n-1, k, m))%m

# Ejecutable
a, b, k, l = getTwo()
print(int(k/ l)* b + min((k % l)* a, b))