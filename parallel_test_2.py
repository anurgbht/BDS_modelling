import multiprocessing as mp
import time
import numpy as np
import random
import os
import pandas as pd
##import my_functions as mf
import pickle

def cube(x):
    print("entering cube")
    return x**3

if __name__ == '__main__':
    import my_functions as mf
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
    output = [p.get() for p in results]
    print(output)
