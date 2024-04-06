import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
        self.optim = optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def backward(self, y_pred, y):
        loss = F.cross_entropy(y_pred, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss


x = [[0, 0], [1, 0], [0, 1], [1, 1]]
y = [[0, 1], [1, 0], [1, 0], [0, 1]]

model = XOR()

for _ in range(4000):
    choice = random.randrange(0, 4)
    xv = torch.Tensor(x[choice])
    yv = torch.Tensor(y[choice])

    pred = model.forward(xv)
    loss = model.backward(pred, yv)

    print(f"Loss: {loss}")

tot = 0
got = 0

for _ in range(50):
    choice = random.randrange(0, 4)
    xv = torch.Tensor(x[choice])
    yv = torch.Tensor(y[choice])

    pred = model.forward(xv)

    parg = np.argmax(pred.detach().numpy())
    pact = np.argmax(yv.detach().numpy())

    tot += 1
    if parg == pact:
        got += 1


print(f"Accuracy: {(got/tot) * 100}")
