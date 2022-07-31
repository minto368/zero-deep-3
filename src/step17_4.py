import numpy as np
import weakref

a = np.array([1, 2, 3])
b = weakref.ref(a)

print(f'before: {b}')
print(b())

a = None
print(f'after: {b}')