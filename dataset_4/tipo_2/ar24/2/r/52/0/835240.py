import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)

MAX_VAL = float('inf')

# --- Input helpers ---
def read_int():
    return int(raw_input())

def read_two_ints():
    return map(int, raw_input().split(" "))

def read_str():
    return raw_input()

def read_strs():
    return raw_input().split(" ")

def read_chars():
    return list(raw_input())

# --- String utility ---
def concatenate(chars):
    return ''.join(chars)

# --- Iteration helpers ---
def permute(seq, r):
    return itertools.permutations(seq, r)

def combine(seq, r):
    return itertools.combinations(seq, r)

# --- Combinatorial modulo ---
def comb_mod(n, k, mod):
    result = 1
    for i in xrange(k):
        result = result * (n - i) % mod
        result = result * inv_mod(i + 1, mod) % mod
    return result

# --- Extended GCD and modular inverse ---
def ext_gcd(a, b):
    x0, x1 = 0, 1
    y0, y1 = 1, 0
    while b != 0:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
        y0, y1 = y1 - q * y0, y0
    return x1, y1, a

def inv_mod(a, m):
    inv, _, _ = ext_gcd(a, m)
    return inv % m

# --- Bisect helper ---
def find_index(arr, val):
    idx = bisect.bisect_left(arr, val)
    if idx != len(arr) and arr[idx] == val:
        return idx
    return -1

# --- Caching decorator ---
def cache_decorator(fn):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    return wrapper

@cache_decorator
def comb_cached(n, k, mod):
    k = min(k, n - k)
    if k < 0 or k > n or n < 0:
        return 0
    if n == 0:
        return 1
    return (comb_cached(n - 1, k - 1, mod) + comb_cached(n - 1, k, mod)) % mod

# --- Main logic ---
x, y, p, q = read_two_ints()
print(int(p / q) * y + min((p % q) * x, y))
