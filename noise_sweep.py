import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from memsim import Net, MemristiveLinear, set_physics, train, test

torch.manual_seed(42)
np.random.seed(42)

def main_sweep():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=1000)

    # 2. Initialize Model
    # Start with 0 noise during training (clean software training)
    model = Net(use_memristors=True, noise_level=0.0) 
    model.to(device)

    # 3.Train
    print(">> Training Model (Phase 1)...")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, device, train_loader, optimizer, epochs=1) 
    
    # Verify baseline accuracy
    set_physics(model, active=False)
    print(">> Baseline Software Accuracy:")
    test(model, device, test_loader, "Software")

    # 4. Sweep
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    print("\n--- Starting Noise Sweep (Hardware Simulation) ---")
    print(f"{'Noise Level':<15} | {'Accuracy':<10}")
    print("-" * 30)

    for noise in noise_levels:
        # A. Update Noise Level
        for m in model.modules():
            if isinstance(m, MemristiveLinear):
                m.noise_scale = noise
                m.static_noise = None # Reset so we generate NEW damage for this level
        
        # B. Test with Physics Enabled
        set_physics(model, active=True)
        
        acc = test(model, device, test_loader, f"Noise {int(noise*100)}%")

if __name__ == "__main__":
    main_sweep()