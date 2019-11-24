from torch import FloatTensor
import torch
from torch.autograd import Variable


# Define the leaf nodes
a = Variable(FloatTensor([4, 5]), requires_grad=True)

weights = [Variable(FloatTensor([i]), requires_grad=True) for i in ([2, 3], [4, 5])]

# unpack the weights for nicer assignment
w1, w2 = weights

b = w1 * a
# a = a[[1, 0]]
a = a.index_select(0, torch.tensor([1, 0], dtype=torch.long))
c = w2 * a + w2
L = (10 - b - c)

# print(a)
# # a = a[[1, 0]]
# a = a.index_select(0, torch.tensor([1, 0], dtype=torch.long))
# print(a)

L.register_hook(lambda grad: print(grad))
b.register_hook(lambda grad: print(grad))
c.register_hook(lambda grad: print(grad))
a.register_hook(lambda grad: print(grad))

L.sum().backward()

print("hier")
print(a.grad)
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(f"Gradient of L w.r.t to w{index}: {gradient}")