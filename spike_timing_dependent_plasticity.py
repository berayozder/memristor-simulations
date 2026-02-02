import numpy as np
import matplotlib.pyplot as plt
from single_memristor import SingleMemristor

def generate_spike(t, trigger_time, amp, tau=0.01):
    # Mimics a biological Action Potential (Shark Fin shape)
    # tau: The decay constant (how fast the voltage fades)
    if t < trigger_time:
        return 0.0
    
    #Calculate the time since the spike
    delta_t = t - trigger_time

    # Check if the spike is "over" (e.g., after 5*tau it's basically 0)
    if delta_t > 5 * tau:
        return 0.0
        
    # Exponential Decay Formula: V(t) = V_peak * e^(-t/tau)
    # We use V_peak = 2.0 Volts to ensure strong reaction
    return amp * np.exp(-delta_t / tau)

def main_stdp():
    # Setup
    dt = 1e-4
    sim_duration = 0.3
    time = np.arange(0, sim_duration, dt)
    
    # Initialize Memristor
    # Initialize Memristor at Middle State
    # (So we can see it go up or down)
    synapse = SingleMemristor(R_on=100, R_off=16000, D=10e-9, mu=10e-14)
    synapse.x = 0.5

    # Define Spike Times (The Event)
    t_pre = 0.10   # Pre-synaptic spike at 100ms
    t_post = 0.11  # Post-synaptic spike at 110ms (10ms delay)

    # Storage
    v_pre_log = []
    v_post_log = []
    state_log = []

    # The loop for STDP
    for t in time:
        # Get voltages from neurons
        v_pre = generate_spike(t, t_pre, amp=2.0)
        v_post = generate_spike(t, t_post, amp=1.5)
        # The voltage across the synapse is the DIFFERENCE
        v_synapse = v_pre - v_post

        # Calculate Current (Ohm's Law)
        # I = V / R
        current = v_synapse / synapse.get_resistance()

        # Update Physics
        synapse.step(current, dt)

        # Log data
        v_pre_log.append(v_pre)
        v_post_log.append(v_post)
        state_log.append(synapse.x)


    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Top: The Neuron Spikes
    plt.subplot(3, 1, 1)
    plt.plot(time, v_pre_log, label="Pre-Synaptic (Input)", color='blue')
    plt.plot(time, v_post_log, label="Post-Synaptic (Output)", color='red', linestyle='--')
    plt.title("The Spike Event: Pre leads Post (Causality)")
    plt.ylabel("Voltage (V)")
    plt.legend(loc="upper right")
    plt.grid(True)
    
    # Middle: The Memristor State (The Learning)
    plt.subplot(3, 1, 2)
    plt.plot(time, state_log, color='green', linewidth=2)
    plt.title("Synaptic Weight Evolution (State 'x')")
    plt.ylabel("Dopant Width (x)")
    plt.grid(True)
    
    # Annotate the result
    start_x = state_log[0]
    end_x = state_log[-1]
    change = end_x - start_x
    result_text = "Potentiation" if change > 0 else "Depression"
    print(f"Start x: {start_x:.4f} | End x: {end_x:.4f} | Change: {change:.4e} ({result_text})")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_stdp()