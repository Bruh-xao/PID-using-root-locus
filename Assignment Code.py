import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# System Parameters (unchanged)
Ja = 0.05     # Motor inertia (kg·m²)
JL = 5        # Load inertia (kg·m²)
Da = 0.01     # Motor damping (N·m·s/rad)
DL = 3        # Load damping (N·m·s/rad)
Ra = 5        # Armature resistance (Ohms)
Kb = 1        # Back EMF constant (V·s/rad)
Kt = 1        # Motor torque constant (N·m/A)
K1 = 150      # Amplifier gain
N1 = 50       # First gear ratio
N2 = 250      # Second gear ratio

# Effective parameters
J_eq = Ja + JL / (N1 * N2)**2
D_eq = Da + DL / (N1 * N2)**2
K_eff = (Kt * K1) / (Ra * N1 * N2)

# Plant transfer function
numerator = [K_eff]
denominator = [J_eq, D_eq, K_eff]
G = ctrl.TransferFunction(numerator, denominator)

# PID Configurations (example values)
configurations = {
    "Config 1": {"Kp": 100, "Ki": 500, "Kd": 40},
    "Config 2": {"Kp": 80, "Ki": 300, "Kd": 50},
    "Config 3": {"Kp": 120, "Ki": 700, "Kd": 30},
}

# Time vector for simulation
t = np.linspace(0, 5, 500)  # 0 to 5 seconds, 500 points

# Root Locus Plot for Open-Loop System (Separate Plot)
plt.figure(figsize=(10, 6))
ctrl.root_locus(G)
plt.title("Root Locus of the Open-Loop System")
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.grid(True)
plt.show()

# Calculate metrics for each configuration and plot step responses
results = {}
for name, params in configurations.items():
    Kp, Ki, Kd = params["Kp"], params["Ki"], params["Kd"]
    C = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    T_pid = ctrl.feedback(C * G, 1)
    
    # Step response
    t_pid, y_pid = ctrl.step_response(T_pid, T=t)
    
    # Ramp response (steady-state velocity error)
    ramp_input = t  # Unit ramp input
    _, y_ramp = ctrl.forced_response(T_pid, T=t, U=ramp_input)
    
    # Maximum Overshoot
    max_overshoot = (np.max(y_pid) - 1) * 100  # in percentage
    
    # Settling Time (2% criteria)
    steady_state = y_pid[-1]
    settling_indices = np.where(np.abs(y_pid - steady_state) > 0.02 * steady_state)[0]
    settling_time = t[settling_indices[-1]] if len(settling_indices) > 0 else 0
    
    # Velocity constant (Kv) from ramp response
    Kv = y_ramp[-1] / t[-1]
    
    # Store results
    results[name] = {
        "Max Overshoot (%)": max_overshoot,
        "Settling Time (s)": settling_time,
        "Kv": Kv,
    }

# Step Response Plot for All Configurations (Separate Plot)
plt.figure(figsize=(10, 6))
for name, params in configurations.items():
    Kp, Ki, Kd = params["Kp"], params["Ki"], params["Kd"]
    C = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    T_pid = ctrl.feedback(C * G, 1)
    t_pid, y_pid = ctrl.step_response(T_pid, T=t)
    plt.plot(t_pid, y_pid, label=f'{name}')
    
# Open-loop system for comparison
t_open, y_open = ctrl.step_response(G, T=t)
plt.plot(t_open, y_open, label='Open-Loop System', color='blue', linestyle='--')

# Customize the step response plot
plt.title("Step Response: PID-Controlled vs Open-Loop Systems")
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Display Results in Console
print("Performance Metrics for Each Configuration:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
