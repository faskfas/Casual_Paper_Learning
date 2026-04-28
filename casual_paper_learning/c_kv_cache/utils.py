import time 
import functools


def timer(func):
    @functools.wraps(func)  # 保留被装饰的函数的属性
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        print(f'Time consuming: {end-start: .6f}s')

        return result
    
    return wrapper  # 可以理解成把原函数包装成了一个新函数
