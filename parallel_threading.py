import multiprocessing as mp
import time
import threading

def cube(x):
##    print("entering cube")
    return x**3

def calc_cube(numbers):
    for n in numbers:
        time.sleep(0.2)
        print(cube(n))

arr = list(range(10))
arr1 = list(range(10))

ts = time.time()
calc_cube(arr)
calc_cube(arr1)
te = time.time()
print("time taken : ",te-ts)

ts = time.time()
t1 = threading.Thread(target=calc_cube,args=(arr,))
t2 = threading.Thread(target=calc_cube,args=(arr1,))
t1.start()
t2.start()
t1.join()
t2.join()

te = time.time()
print("time taken : ",te-ts)

