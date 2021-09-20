import torch
import numpy as np

torch.set_printoptions(threshold=np.inf)

a = torch.tensor(np.arange(1, 3 * 4 * 5 * 6 + 1)).reshape(1, 3, 4, 5, 6).float()
w = torch.ones(17*3*3*2*1).float()
offset = 0.01
for elem in w:
    elem += offset
    offset += 0.01


w = w.reshape(17,3,3,2,1)
out = torch.conv3d(input=a, weight=w, stride=1, padding=0, dilation=(1,2,3))

print(out.shape, out.reshape(17, -1))