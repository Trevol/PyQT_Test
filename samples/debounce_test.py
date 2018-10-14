from utils import debounce
import time
import threading

counter = 0

print(f'Main - 1: {threading.current_thread().ident}')

@debounce(.5)
def test():
    print(f'test: {threading.current_thread().ident}')
    global counter
    counter += 1

test()
test()
test()

time.sleep(.2)

test()

print(f'Main - 2: {threading.current_thread().ident}')
