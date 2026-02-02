import torch
import torch.nn as nn
import torch.nn.functional as F

class MemristiveLinear(nn.Linear):
    """
    A Linear layer that simulates the manufacturing variability of memristor crossbars.
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        noise_scale (float): The standard deviation of the static write noise.
    """
    def __init__(self, in_features, out_features, noise_scale=0.1, bias=True):
        super(MemristiveLinear, self).__init__(in_features, out_features, bias)
        self.noise_scale = noise_scale
        self.enable_physics = False
        self.static_noise = None # To store the permanent damage

    def program_weights(self):
        """Simulates the one-time 'Programming' of the chip."""
        # We generate the noise ONCE and store it
        self.static_noise = torch.randn_like(self.weight) * self.noise_scale * torch.abs(self.weight)

    def forward(self, input):
        # Only inject noise if the user explicitly asks for it
        if self.enable_physics:
            # If we haven't programmed the chip yet, do it now
            if self.static_noise is None:
                self.program_weights()
                
            # Use the SAME frozen noise every time (Write Error)
            noisy_weight = self.weight + self.static_noise
            return F.linear(input, noisy_weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

def set_physics(model, active=True):
    for module in model.modules():
        if isinstance(module, MemristiveLinear):
            module.enable_physics = active
