#%%
import numpy as np
#%%
# --- Input Parameters ---
D50 = 2e-4          # Median grain size [m]
D90 = 2e-4          # [m]
rho_s = 2650        # Sediment density [kg/m3]
rho_w = 1000        # Water density [kg/m3]
d = 2.875           # Water depth [m] (average of 2.75 and 3)
zeta = 0            # Bed level offset [m] (assumed 0)
theta_cr = 0.047    # Critical Shields parameter
xi = 1.0            # Hiding/exposure factor (assumed 1 for uniform grain size)
alpha = 1.0         # Calibration coefficient
g = 9.81            # Gravity [m/s2]

# --- Calculations ---
# 1. Relative density
Delta = (rho_s - rho_w) / rho_w

# 2. Chézy coefficient related to grains (C_g,90)
# Equation 8.211: 18 * log10(12 * (d + zeta) / D90)
C_g90 = 18 * np.log10(12 * (d + zeta) / D90)

# 3. Chézy coefficient for the total flow (C)
# Note: If we assume a flat bed for the threshold of motion, C = C_g90.
# This makes the ripple factor mu = 1.0.
C = C_g90 
mu = min((C / C_g90)**1.5, 1.0)

# 4. Solving for critical velocity q
# At the threshold: mu * theta = xi * theta_cr
# From Eq 8.209: theta = (q / C)**2 * (1 / (Delta * D50))
# Therefore: mu * (q_crit / C)**2 * (1 / (Delta * D50)) = xi * theta_cr

q_crit = np.sqrt((xi * theta_cr * Delta * D50 * C**2) / mu)

print(f"--- MPM Critical Threshold Analysis ---")
print(f"Relative Density (Delta): {Delta:.3f}")
print(f"Grain Chezy (C_g90):      {C_g90:.2f} m^0.5/s")
print(f"Ripple Factor (mu):       {mu:.2f}")
print(f"---------------------------------------")
# %%

print(f"Critical Velocity (q):    {q_crit:.4f} m/s")
# %%
import numpy as np

# --- Inputs ---
D50 = 2e-4
D90 = 2e-4
rho_s = 2650
rho_w = 1000
g = 9.81
d = 2.875  # Average depth
theta_cr = 0.047
width = 200 # ASSUMPTION: Example channel width in meters to convert Q to q

# Derived Constants
Delta = (rho_s - rho_w) / rho_w
C_g90 = 18 * np.log10(12 * d / D90)

# Calculate Critical Velocity (q_crit) assuming mu=1 at threshold
# From mu * theta = theta_cr -> q_crit = sqrt(theta_cr * Delta * D50 * C_g90**2)
q_crit = np.sqrt(theta_cr * Delta * D50 * C_g90**2)

# --- Scenario Check ---
Q_means = [500, 1000]

print(f"{'Mean Q':<10} | {'Mean Velocity (q)':<20} | {'Status'}")
print("-" * 50)

for Q in Q_means:
    v_mean = Q / (width * d) # Flow velocity u = Q/A
    status = "ACTIVE" if v_mean > q_crit else "NEAR/BELOW THRESHOLD"
    print(f"{Q:<10} | {v_mean:<20.4f} | {status}")

print(f"\nCritical Velocity Threshold (q_crit): {q_crit:.4f} m/s")
# %%
def calculate_S(q, d=2.875):
    # Constants from your setup
    C_g90 = 18 * np.log10(12 * d / D90)
    theta = (q / C_g90)**2 / (Delta * D50)
    
    # Simple MPM transport proportional part (ignoring constants for scaling check)
    if theta > theta_cr:
        return (theta - theta_cr)**1.5
    return 0

# Check sensitivity (slope) at Q=500 and Q=1000 velocities
q_500 = 0.8696
q_1000 = 1.7391

# Finite difference to find sensitivity
dq = 0.01
S_low = (calculate_S(q_500 + dq) - calculate_S(q_500)) / dq
S_high = (calculate_S(q_1000 + dq) - calculate_S(q_1000)) / dq

print(f"Transport sensitivity at Q=500:  {S_low:.4f}")
print(f"Transport sensitivity at Q=1000: {S_high:.4f}")
print(f"Ratio: {S_high/S_low:.2f}x more sensitive at high flow")

# %%
