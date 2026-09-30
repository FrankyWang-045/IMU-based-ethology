import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_samples= 100
data = torch.randn(n_samples,2)
labels = (data[:,0]**2 + data[:,1]**2 < 1).float().unsqueeze(1)  # Points inside the unit circle are labeled as 1, outside as 0

plt.scatter(data[:,0], data[:,1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

class SimpleNN(nn.Module):
    def __init__(self):
        super(Simple)