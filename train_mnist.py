import torch
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os

from memsim import Net, set_physics, train, test

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Download MNIST
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=1000)

    # --- STEP 1: Train the "Ideal" Software Model ---
    # We use our custom class, but noise is only applied during eval()
    model = Net(use_memristors=True, noise_level=0.15) # 15% Device Variation (High noise!)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, device, train_loader, optimizer)

    # --- STEP 2: Compare Software vs Hardware ---
    
    # A. Test in "Training Mode" (Clean Software Weights)
    model.train() 
    set_physics(model, active=False) # Disable noise
    acc_soft = test(model, device, test_loader, "Software Accuracy (Ideal)")
    
    # B. Test in "Eval Mode" (Noisy Hardware Weights)
    model.eval()
    set_physics(model, active=True) # Enable noise
    acc_hard = test(model, device, test_loader, "Hardware Accuracy (Simulated)")

    print(f"\n Impact of Memristor Noise (15%): {acc_soft - acc_hard:.2f}% Accuracy Drop")

if __name__ == "__main__":
    main()
