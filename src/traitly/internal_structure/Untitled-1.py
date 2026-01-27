# Source - https://stackoverflow.com/a
# Posted by unutbu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-26, License - CC BY-SA 4.0

import timeit
counter = 0
reverse_counter = 0


setup = '''\
import random
x = [random.random() for i in range(10**6)]
y = [random.random() for i in range(10**6)]
'''    
multiply = '[xi*yi for xi, yi in zip(x, y)]'
divide = '[xi/yi for xi, yi in zip(x, y)]'

N = 100
for i in range(0, N):
    n = timeit.timeit(multiply, setup=setup, number=3)
    m = timeit.timeit(divide, setup=setup, number=3)    

    if n < m:
        counter = counter + 1

print('multiply is faster {:.2%} of the time'.format(counter/N))
