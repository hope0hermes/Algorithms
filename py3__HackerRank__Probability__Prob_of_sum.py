#!/usr/bin/python3
#encoding-utf8

"""
    Random number generator

    There is an ideal random number generator, which given a positive integer M
    can generate any real number between 0 to M, and probability density
    function is uniform in [0, M].

    Given two numbers A and B and we generate x and y using the random number
    generator with uniform probability density function [0, A] and [0, B]
    respectively, what's the probability that x + y is less than C? where C is a
    positive integer.

    Input Format

    The first line of the input is an integer N, the number of test cases.

    N lines follow. Each line contains 3 positive integers A, B and C.

    Constraints

    All the integers are no larger than 10000.

    Output Format

    For each output, output a fraction that indicates the probability. The
    greatest common divisor of each pair of numerator and denominator should be
    1.
"""

def main():
    # Total number of tests.
    # N = int(input('N -> ').split()[0])
    N = int(input().split()[0])
    # Parse each test.
    numer = 0
    denom = 1
    res = []
    for idx in range(N):
        # line = [int(x) for x in input('A B C -> ').split()]
        line = [int(x) for x in input().split()]
        suma = line[0] + line[1]
        numer = line[2]**2
        denom = 2*line[0]*line[1]
        if(numer > denom):
            res.append([1,1])
        else:
            norm = gcd(numer, denom)
            # Normalize.
            numer /= norm
            denom /= norm
            res.append([numer,denom])
    for idx in range(N):
        print('{:d}/{:d}'.format(int(res[idx][0]), int(res[idx][1])))


if __name__ == '__main__':
    from sys import argv
    from fractions import gcd

    main()
