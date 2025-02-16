#PART 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (ensure you have a CSV file with Year, GDP, Capital, Labor columns)
data = pd.read_csv("India_Data.csv")
# Parameters
alpha = 0.35  # Capital share assumption

# Compute log values
data["log_GDP"] = np.log(data["GDP"])
data["log_Capital"] = np.log(data["Capital_Stock"])
data["log_Labor"] = np.log(data["Annual_Hours"])

# Compute Solow residual (log A_t)
data["log_At"] = data["log_GDP"] - (alpha * data["log_Capital"]) - ((1 - alpha) * data["log_Labor"])
print()
# Convert back to levels
data["At"] = np.exp(data["log_At"])

# Convert "Year" column safely
data["Year"] = pd.to_datetime(data["Year"], errors="coerce").dt.year  # Extract year
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")  # Ensure numeric

# Drop missing values if conversion failed
data = data.dropna(subset=["Year"])

plt.figure(figsize=(10, 5))

# Plot Solow Residual
plt.plot(data["Year"], data["At"], marker="o", linestyle="-", color="green", label="Solow Residual (A_t)")

# Set X-Ticks with 10-year intervals
plt.xticks(
    ticks=list(range(int(data["Year"].min()), int(data["Year"].max()) + 1, 5)),
    rotation=45,
    fontsize=12
)

plt.xlabel("Year")
plt.ylabel("Solow Residual (A_t)")
plt.title("Solow Residual for India")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()

# Save the processed data
data.to_csv("india_growth_processed.csv", index=False)

#PART 2
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data.dropna(subset=["Year"], inplace=True)

# Extract necessary variables
years = data["Year"].values
A_t = data["At"].values  # Use actual Solow Residual
K_t = data["Capital_Stock"].values  # Capital Stock
L_t = data["Annual_Hours"].values  # Labor Force

# Parameters
s = 0.2927  # Savings rate (29.27%)
delta = 0.07  # Depreciation rate (7%)
alpha = 0.33  # Capital elasticity

# Set base year to 1991
base_year = 1991
base_index = np.where(years == base_year)[0][0]

# Compute initial capital per worker
k0 = K_t[base_index] / L_t[base_index]

# Time periods for simulation
t_max = len(years)
k_t = np.zeros(t_max)
k_t[0] = k0


# Compute path of k_t using actual A_t values
def solow_growth(k, s, A, alpha, delta):
    return s * A * k ** alpha + (1 - delta) * k


for t in range(1, t_max):
    k_t[t] = solow_growth(k_t[t - 1], s, A_t[t], alpha, delta)

    # Check for steady state (convergence)
    if abs(k_t[t] - k_t[t - 1]) < 1e-3:
        print(f"Steady state reached at period {years[t]} with k* = {k_t[t]:.2f}")
        k_t = k_t[:t + 1]  # Trim the array to the converged period
        years = years[:t + 1]
        break

# Debugging print statements
print("First few values of k_t:", k_t[:5])
print("Final value of k_t (steady state):", k_t[-1])

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(years, k_t, label="Capital per Worker (k_t)", color='b')
plt.axhline(y=k_t[-1], color='r', linestyle='--', label="Steady State k*")
plt.xlabel("Year")
plt.ylabel("Capital per Worker (k_t)")
# plt.yscale("log")  # Apply logarithmic scale to y-axis
plt.title("Evolution of Capital per Worker in India")
plt.legend()
plt.grid()
plt.show()
#Part 2b
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (Ensure the file name is correct)
data = pd.read_csv("india_growth_processed.csv")  # Change filename if needed

# Ensure the "Year" column is numeric and sorted
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data = data.dropna(subset=["Year"]).sort_values(by="Year")

# Extract relevant data
years = data["Year"].values
A_t = data["At"].values  # Extract Solow residual values
L_t = data["Annual_Hours"].values  # Extract labor supply (total annual hours worked)
GDP_actual = data["GDP"].values  # Extract actual GDP data

# Define capital elasticity parameter
alpha = 0.33

# Compute model-based aggregate output Y_t
Y_t = A_t * (L_t ** (1 - alpha))

# Adjust scale for better visualization
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, Y_t, label="Model-Based Aggregate Output (Y_t)", color='blue', linewidth=2)
ax.plot(years, GDP_actual, label="Actual GDP", color='red', linestyle='dashed', linewidth=2)

ax.set_xlabel("Year")
ax.set_ylabel("Aggregate Output / GDP in trillions")
ax.set_title("Comparison of Model-Based Output and Actual GDP in India")
ax.legend()
ax.grid(True)

# Adjust y-axis scale if necessary
ax.set_ylim([0, max(max(Y_t), max(GDP_actual)) * 1.1])  # Adding 10% buffer

plt.show()

#ALtering Parameters and analyzing model parameters and productivity influence on evolution of k_t
# Set base year to 1991
base_year = 1991
base_index = np.where(years == base_year)[0][0]

# Compute initial capital per worker
k0 = K_t[base_index] / L_t[base_index]

# Time periods for simulation
t_max = len(years)


# Solow growth function
def solow_growth(k, s, A, alpha, delta):
    return s * A * k ** alpha + (1 - delta) * k


# Different parameter sets for sensitivity analysis
parameter_sets = [
    (0.2927, 0.07, 0.33),  # Baseline
    (0.2927, 0.07, 0.4),   # Higher capital elasticity
    (0.2927, 0.1, 0.33),   # Higher depreciation
    (0.2927, 0.1, 0.4),    # Higher depreciation & capital elasticity
    (0.35, 0.07, 0.33),    # Higher savings
    (0.35, 0.07, 0.4),     # Higher savings & capital elasticity
    (0.35, 0.1, 0.33),     # Higher savings & depreciation
    (0.35, 0.1, 0.4)       # Higher savings, depreciation & capital elasticity
]

# Initialize a figure
plt.figure(figsize=(10, 5))

# Loop through each parameter set
for (s, delta, alpha) in parameter_sets:
    k_t = np.zeros(t_max)
    k_t[0] = k0

    for t in range(1, t_max):
        k_t[t] = solow_growth(k_t[t - 1], s, A_t[t], alpha, delta)

        # Check for steady state (convergence)
        if abs(k_t[t] - k_t[t - 1]) < 1e-3:
            k_t = k_t[:t + 1]  # Trim the array to the converged period
            years = years[:t + 1]
            break

    # Plot capital per worker in log scale
    plt.plot(years, np.log(k_t), label=f's={s}, δ={delta}, α={alpha}')

# Labeling and plot settings
plt.xlabel("Year")
plt.ylabel("Capital per Worker k_t)")
plt.title("Effect of Parameter Changes on Capital per Worker Evolution")
plt.legend()
plt.grid()
plt.show()

#PART 3
# Calculate the steady-state output and capital for each year
Y_t_steady = np.zeros(len(years))  # Steady-state output (Y_t,steady)
K_t_steady = np.zeros(len(years))  # Steady-state capital (K_t,steady)

for t in range(len(years)):
    # Calculate steady-state capital (K_t,steady) for each year
    K_t_steady[t] = (s * A_t[t] / delta) ** (1 / (1 - alpha)) * L_t[t]

    # Calculate steady-state output (Y_t,steady) for each year
    Y_t_steady[t] = A_t[t] * K_t_steady[t] ** alpha

# Check if GDP values are large, and scale them down for visualization
GDP_actual_scaled = GDP_actual / 1e9  # Example: divide by 1 billion

# Plot the results with scaled GDP
plt.figure(figsize=(10, 6))
plt.plot(years, GDP_actual, label="Actual GDP (Y_t)", color='red', linestyle='--', linewidth=2)
plt.plot(years, Y_t_steady, label="Steady-State Output (Y_t,steady)", color='green', linestyle='-.', linewidth=2)

plt.xlabel("Year")
plt.ylabel("Output (Y_t) in Trillions")
plt.title("Comparison of Steady-State Output and Actual GDP")
plt.legend()
plt.grid(True)
plt.show()
