import numpy as np
import math
import matplotlib.pyplot as plt

def function_1(x):
    return x**2+1

"""
xs = np.linspace(1, 10, 10)

for x in xs:
    print(function_1(x))
"""

def minimise(func, a, b, tol, verbose):
    """
    This function finds the minimum of the given one dimension
    function between assigned internal [a, b]
    :param func: the target one dimension function
    :param a: beginning of the internal
    :param b: end of the internal
    :param tol: the number of points within internal or precision
    :param verbose: parameter determining whether return output
    :return: the minimum within internal and corresponding x;
    """
    loc = []                        # store all minimumu location
    results = []                    # store all results for drawing

    xs = np.linspace(a, b, tol)     # create interval data under assigned precision
    min = func(a)                   # initialize min

    # travel all x to find minimum; replace locations when new minimum occurs and
    # append new location if minimum is repeated
    for x in xs:
        results.append(func(x))
        if func(x) < min:
            min = func(x)
            loc.clear()
            loc.append(x)
        elif func(x) == min:
            loc.append(x)

    # drawing figure for observation
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(xs, results)
    plt.show()

    # using verbose to control output
    if verbose:
        print("Minimum: {}; Corresponding x location: {}".format(min, loc))
        return min, loc
    else:
        print("Calculation complete.")
        return min, loc


verbose = True
min, loc = minimise(function_1, -20, 20, 1000, verbose)


