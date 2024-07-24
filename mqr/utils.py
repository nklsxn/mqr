import numpy as np

def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation

    Thanks to: https://stackoverflow.com/a/50992575

    Arguments
    ---------
    n -- the number to express in ordinal representation
    '''
    if not np.isclose(n, int(n)):
        number = str(n)
        suffix = 'th'
    elif (11 <= (n % 100) <= 13) or ():
        number = str(int(n))
        suffix = 'th'
    else:
        number = str(int(n))
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(int(n) % 10, 4)]
    return number + suffix
