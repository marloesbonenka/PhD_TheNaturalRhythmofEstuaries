"This is a quick testing function for the components the .bct file consists of"
#%%
import matplotlib.pyplot as plt
import numpy as np
import datetime

#%%

# Time axis
t = np.linspace(0, 365, 1000)
T = 365.25
num_steps = len(t)

# Convert day-of-year to datetime objects (assuming year = 2000 for simplicity)
base_date = datetime.datetime(2024, 1, 1)
dates = [base_date + datetime.timedelta(days=day) for day in t]

# Seasonal components
Q_spring = 0.25 * np.sin(2 * np.pi * (t - 30) / T)
Q_autumn = 0.15 * np.sin(2 * np.pi * (t - 300) / T)
Q_seasonal = Q_spring + Q_autumn

# Summer rainfall event (Gaussian)
A_summer = 0.35
t_summer = 200
sigma_summer = 5
Q_summer = A_summer * np.exp(-0.5 * ((t - t_summer) / sigma_summer)**2)

#%% Combine seasonal + summer
Q0 = 1  # Assume mean discharge = 1 for visualization
Q_mean = Q0 * (1 + Q_seasonal + Q_summer)

# Generate AR(1) noise
cv = 0.3               # coefficient of variation
ar1_rho = 0.95         # autocorrelation
ar1_std = cv * Q0
rng = np.random.default_rng(0)
noise = np.zeros(num_steps)
noise[0] = rng.normal(0, ar1_std)
for i in range(1, num_steps):
    noise[i] = ar1_rho * noise[i - 1] + rng.normal(0, ar1_std * np.sqrt(1 - ar1_rho**2))

# Add noise to the seasonal signal
Q_with_noise = Q_mean + noise
Q_with_noise[Q_with_noise < 0] = 0  # ensure positivity

if abs(np.mean(Q_with_noise) - Q0) > 0.01 * Q0:
    print('Deviation is larger than 1% -- Q_with_noise =', f'{np.mean(Q_with_noise):.2f}', 'Q0 =', Q0)
    print('Absolute difference =', f'{abs(np.mean(Q_with_noise) - Q0):.2f}')

    Q_with_noise = Q_with_noise * (Q0 / np.mean(Q_with_noise))

    print('After correction, Q_with_noise =', f'{np.mean(Q_with_noise):.2f}')

Q_with_noise[Q_with_noise < 0] = 0  # ensure positivity

#%%

# Plotting
plt.figure(figsize=(10, 5))

# Without noise
plt.plot(dates, Q_mean, label='Without AR(1) noise', color='steelblue')
plt.axhline(np.mean(Q_mean), label = 'mean', linestyle = 'dashed', color = 'grey')
plt.xlabel("Day of year")
plt.ylabel("Discharge [relative]")
plt.title("Seasonal signal without noise")
plt.grid(True)
plt.legend(loc='upper right')

# With noise
plt.figure(figsize=(10, 5))
plt.plot(dates, Q_with_noise, label='With AR(1) noise', color='darkorange')
plt.axhline(np.mean(Q_with_noise), label = 'mean', linestyle = 'dashed', color = 'grey')
plt.xlabel("Day of year")
plt.title("Seasonal signal with AR(1) noise")
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
# %%
