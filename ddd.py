import numpy as np
import torch

rr1 = np.array([0,7])
cc1 = np.array([0,7])
a = np.zeros((8,8), dtype=np.uint8)
a[rr1,cc1] = 1
print(a)
#canvas = np.zeros((4,4))
#canvas[rr1, cc1] += 1
# if np.any(rr1>2):
#     print('rrrrrrrrrrrrrrr1', rr1)
# else:
# #     print('zzzzzz')
# if np.any(rr1 > 2):
#     print('rrrrrrrrrrrrrrr1>2')
#     rrr1 = np.where(rr1 > 2, 0, rr1)
#     print(rrr1)
# if np.any(cc1 > 1):
#     print('ccccccccccccccc1>1')
#     ccc1 = np.where(cc1 > 1, 0, cc1)
#     print(ccc1)
# a=[1,2]
# if len(a) == 0:
#     print("this tensor is empty")
# else:
#     print("this tensor is not empty")
# b = a[-1]
# print(a)
# print(b)


