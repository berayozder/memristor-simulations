import numpy as np
import matplotlib.pyplot as plt
class SingleMemristor:
    def __init__(self,R_on,R_off,D,mu):
        # Physical constants
        self.R_on = R_on    # Resistance when fully doped (low resistance)
        self.R_off = R_off  # Resistance when undoped (high resistance)
        self.D = D          # Thickness of the device
        self.mu = mu        # Mobility of the oxygen vacancies
        self.k = mu * R_on / (D**2) # Constant from Strukov's paper
        # State var
        # x represents how "open" the channel is (0 to 1)
        self.x = 0.1
        # When x is near 0, the memristor is in the high resistance state (R_off)
        # When x is near 1, the memristor is in the low resistance state (R_on)
        
        
    def get_resistance(self):
        # Return the resistance based on current state 'x'
        R_current = self.x * self.R_on + (1 - self.x )* self.R_off
        return R_current
    
    def step(self,current, dt):
        # Update the state 'x' based on the current flowing through
        dx_dt = (self.k) * current #Ion drift
        self.x = self.x + (dx_dt * dt) # The memory (x=∫dx)
        
        # Apply "Hard Boundaries" (if x > 1, set x = 1) 
        # This prevents the simulation from breaking (Strukov's bug)
        # The ions cannot fly out of the device
        if self.x > 1 : self.x = 1
        if self.x < 0 : self.x = 0
        
        voltage = current * self.get_resistance()
        return voltage
    

def main():
    # Setup time
    dt = 1e-4                       # Small time step (0.1 ms)
    T_max = 5.0                     # Simulate for 2 seconds
    time = np.arange(0,T_max,dt)   # Time vector
    
    #Setup Input Source
    freq = 0.5
    amplitude = 400e-6              # 200 micro-Amps 
    current_source = amplitude*np.sin(2*np.pi*freq*time)
   
    #Instantiate the Device 
    memristor_device = SingleMemristor(R_on=100,R_off=16000,D=10*10e-9,mu=10e-14)
    
    
    voltage_samples_read = []
    state_samples_read = []
    
    for i in range(len(time)):
        I_now = current_source[i]
        V_now = memristor_device.step(I_now,dt)
        voltage_samples_read.append(V_now)
        state_samples_read.append(memristor_device.x)
        
    #Plotting
    
    
    # Plot 1: The Famous Hysteresis Loop
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(voltage_samples_read, current_source)
    plt.title("Pinched Hysteresis Loop (V-I)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.grid(True)
    
    # Plot 2: State Evolution
    plt.subplot(1, 2, 2)
    plt.plot(time, state_samples_read)
    plt.title("Internal State (x) over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Dopant Width (normalized)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
    
        
        
    
    

    
    
        
    