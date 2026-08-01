import numpy as np
import torch

a = []

a.append(torch.tensor([1]))
a.append(torch.tensor([2]))

a = np.array(a)
print(type(a[0][0]))