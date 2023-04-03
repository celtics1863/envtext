
import time
def timing(f):
    def fun(*args,**kwargs):
        start = time.time()
        res = f(*args,**kwargs)
        end = time.time()
        print(end-start)
        return res
    return fun
