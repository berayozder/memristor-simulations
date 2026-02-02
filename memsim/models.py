import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MemristiveLinear

# ==========================================
# The Model Architecture
# ==========================================
class Net(nn.Module):
    def __init__(self, use_memristors=False, noise_level=0.0):
        super(Net, self).__init__()
        # We switch between Standard Linear and Memristive Linear
        LayerClass = MemristiveLinear if use_memristors else nn.Linear
        kwargs = {'noise_scale': noise_level} if use_memristors else {}
        
        self.fc1 = LayerClass(784, 256, **kwargs)
        self.fc2 = LayerClass(256, 10, **kwargs)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, epochs=1):
    model.train()
    print(">> Training Software Model...")
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    print(">> Training Complete.")

def test(model, device, test_loader, label):
    model.eval() # This triggers the "Hardware Simulation" logic in our custom layer
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print(f'{label}: {acc:.2f}%')
    return acc
