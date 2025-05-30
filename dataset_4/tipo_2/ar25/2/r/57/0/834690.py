import sys
import math
import itertools
from bisect import bisect_left

sys.setrecursionlimit(10000)

MAX_INF = float('inf')

# Input utilities

def read_number():
    return int(raw_input())

def read_pair():
    return map(int, raw_input().split())

def read_text():
    return raw_input()

def read_tokens():
    return raw_input().split()

def read_char_list():
    return list(raw_input())

# String join helper
def concat_chars(lst):
    return ''.join(lst)

# Iterable generators
def permutations_of(seq, r):
    return itertools.permutations(seq, r)

def combinations_of(seq, r):
    return itertools.combinations(seq, r)

# Modular combinatorics
def mod_comb(n, k, mod):
    result = 1
    for idx in xrange(k):
        result = result * (n - idx) % mod
        result = result * modular_inverse(idx + 1, mod) % mod
    return result

# Extended Euclidean algorithm and inverse

def extended_gcd(u, v):
    x0, x1 = 0, 1
    y0, y1 = 1, 0
    while v != 0:
        q = u // v
        u, v = v, u % v
        x0, x1 = x1 - q * x0, x0
        y0, y1 = y1 - q * y0, y0
    return (x1, y1, u)


def modular_inverse(a, mod):
    inv, _, _ = extended_gcd(a, mod)
    return inv % mod

# Bisect helper to find exact match
def find_pos(arr, value):
    idx = bisect_left(arr, value)
    if idx != len(arr) and arr[idx] == value:
        return idx
    return -1

# Memoization decorator
def cache_func(fn):
    memo = {}
    def inner(*args):
        if args not in memo:
            memo[args] = fn(*args)
        return memo[args]
    return inner
