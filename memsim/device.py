import numpy as np

class SingleMemristor:
    def __init__(self, R_on, R_off, D, mu):
        # Physical constants
        self.R_on = R_on    # Resistance when fully doped (low resistance)
        self.R_off = R_off  # Resistance when undoped (high resistance)
        self.D = D          # Thickness of the device
        self.mu = mu        # Mobility of the oxygen vacancies
        self.k = mu * R_on / (D**2) # Constant from Strukov's paper
        # State var
        # x represents how "open" the channel is (0 to 1)
        self.x = 0.5
        # When x is near 0, the memristor is in the high resistance state (R_off)
        # When x is near 1, the memristor is in the low resistance state (R_on)
        
    def get_resistance(self):
        # Return the resistance based on current state 'x'
        R_current = self.x * self.R_on + (1 - self.x )* self.R_off
        return R_current
    
    def step(self, current, dt):
        # Update the state 'x' based on the current flowing through
        dx_dt = (self.k) * current # Ion drift
        self.x = self.x + (dx_dt * dt) # The memory (x=∫dx)
        
        # Apply "Hard Boundaries" (if x > 1, set x = 1) 
        # This prevents the simulation from breaking (Strukov's bug)
        # The ions cannot fly out of the device
        if self.x > 1 : self.x = 1
        if self.x < 0 : self.x = 0
        
        voltage = current * self.get_resistance()
        return voltage
