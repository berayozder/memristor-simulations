# Memristor Simulations

A collection of simulations exploring memristor behavior and applications in neuromorphic computing.

## 🧠 The Physics (Strukov Model)

The device is modeled as a TiO₂ thin film with oxygen vacancies drifting under an electric field:

**Voltage-current relationship:**
```
v(t) = [R_on · x(t) + R_off · (1 - x(t))] · i(t)
```

**State evolution:**
```
dx/dt = k · i(t) · f(x)
```

Where:
- `x(t)` is the normalized state variable (0 ≤ x ≤ 1) representing the doped region width
- `R_on` is the resistance when fully doped
- `R_off` is the resistance when undoped
- `k` is the drift velocity coefficient
- `f(x)` is a window function preventing state saturation
- `i(t)` is the applied current
- `v(t)` is the resulting voltage

The window function typically takes the form:
```
f(x) = 1 - (2x - 1)^(2p)
```

where `p` controls the nonlinearity (higher p → sharper boundaries).

## 🔬 Project Overview

This hardware-aware simulation explores Memristive Nanodevices and their application in Neuromorphic Computing. The project models the nonlinear dopant drift kinetics (Strukov model) and demonstrates biological learning via Spike-Timing-Dependent Plasticity (STDP).

**Phase 1: Device Physics**  
Simulation of the HP TiO₂ Memristor model, demonstrating the signature "Pinched Hysteresis Loop."

**Phase 2: Synaptic Plasticity**  
Implementation of a memristive synapse connecting Pre- and Post-synaptic neurons.

**Key Finding:**  
Demonstrated that symmetric square-wave spikes fail to induce Long-Term Potentiation (LTP). Implemented exponential decay (shark-fin) spikes and amplitude asymmetry to achieve stable synaptic weight updates.

## 📊 Results

### 1. The Pinched Hysteresis

Evidence of the memristor's non-volatile memory property.

### 2. STDP Learning Window

Demonstration of Long-Term Potentiation (LTP) where the synaptic weight (x) increases and stabilizes.