import bisect
import sys
import math
import itertools
sys.setrecursionlimit(10000)

FIN = float('inf')

# input macro
def iF():
    return int(raw_input())
def iiF():
    return map(int,raw_input().split(" "))
def sF():
    return raw_input()
def ssF():
    return raw_input().split(" ")
def slistF():
    return list(raw_input())
#

def joinF(s):
    return ''.joinF(s)

#iterate macro
def piterF(n,m):
    return itertools.permutations(n,m)
def citerF(n,m):
    return itertools.combinations(n,m)

#modulo macro
def modcF(a,b,m):
    c = 1
    for i in xrange(b):
        c = c * (a - i) % m
        c = c * modinvF(i + 1,m) % m
    return c
 
def gcdF(a, b):
    (x, lastx) = (0, 1)
    (y, lasty) = (1, 0)
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lastx) = (lastx - q * x, x)
        (y, lasty) = (lasty - q * y, y)
    return (lastx, lasty, a)
 
def modinvF(a, m):
    (inv, q, gcd_val) = gcdF(a, m)
    return inv % m

#bisect macro
def FAINAA(a, x):
    #Locate the leftmost value exactly equal to x
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

#memoize macro
def memoizeF(f):
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[(args)] = f(*args)
        return cache[args]
    return helper

@memoize
def ncFk(a,b,m):
    b = min([a-b,b])
    if (b>a or b<0 or a<0):
        return 0
    elif a == 0:
        return 1
    return (ncFk(a-1,b-1,m)+ncFk(a-1,b,m)) % m

###########

a,b,k,l = iiF()
print(int(k/l)*b+min([(k%l)*a,b]))
