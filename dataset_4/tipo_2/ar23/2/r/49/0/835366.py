import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)

INF = float('inf')

# input macro
def a():
    return int(raw_input())
def aa():
    return map(int,raw_input().split(" "))
def b():
    return raw_input()
def bb():
    return raw_input().split(" ")
def blist():
    return list(raw_input())
#

def merge(s):
    return ''.join(s)

#iterate macro
def titer(n,m):
    return itertools.permutations(n,m)
def miter(n,m):
    return itertools.combinations(n,m)

#modulo macro
def divc(a,b,m):
    c = 1
    for i in xrange(b):
        c = c * (a - i) % m
        c = c * popinv(i + 1,m) % m
    return c
 
def dcg(a, b):
    (x, lastx) = (0, 1)
    (y, lasty) = (1, 0)
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lastx) = (lastx - q * x, x)
        (y, lasty) = (lasty - q * y, y)
    return (lastx, lasty, a)
 
def popinv(a, m):
    (inv, q, gcd_val) = dcg(a, m)
    return inv % m

#bisect macro
def main(a, x):
    #Locate the leftmost value exactly equal to x
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

#memoize macro
def memoize(f):
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[(args)] = f(*args)
        return cache[args]
    return helper

@memoize
def knc(a,b,m):
    b = min([a-b,b])
    if (b>a or b<0 or a<0):
        return 0
    elif a == 0:
        return 1
    return (knc(a-1,b-1,m)+knc(a-1,b,m)) % m
