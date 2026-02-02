import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path so we can import memsim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memsim.device import SingleMemristor

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
    
    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

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
    plt.savefig(os.path.join(results_dir, "pinched_hysteresis_loop.png"))
    print(f"Plot saved to {os.path.join(results_dir, 'pinched_hysteresis_loop.png')}")
    # plt.show() # Commented out for batch execution
    
if __name__ == "__main__":
    main()
