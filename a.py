#计算余弦相似度
import torch
import numpy
import numpy as np

a = [[1.0,4.0],[10.0,3.0],[3.0,8.0],[2.0,2.0]]
a = numpy.array([[i] for i in a])
b = a[1]-a[3] #b=[8,1] 从原点开始的
b = torch.tensor(b)
c = a[0]-a[2] #c=[-2,-4]
c = torch.tensor(c)
similar = torch.cosine_similarity(b,c)
print(b,c,similar)
