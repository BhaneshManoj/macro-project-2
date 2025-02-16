# MACRO PROJECT 2
This project builds on the  understanding of growth theory by introducing key components  of the Solow Growth Model and intertemporal optimization. We  calculate the Solow  residual, calibrate and simulate the Solow model, and solve an optimization problem. The  project emphasizes connecting theoretical models to real-world data.

# Solow Growth Model Analysis

## Project Overview
This project involves analyzing economic growth using the Solow Growth Model and solving an intertemporal optimization problem. The analysis includes:
- Calculating the Solow residual
- Calibrating and simulating the Solow model
- Solving a constrained optimization problem

The project connects theoretical models to real-world data, particularly focusing on India using data from the FRED website.

---

## File Descriptions

### 📜 Scripts
- **`prefinal.py`**
  - **Part 1**: Growth decomposition and Solow residual calculation.
  - **Part 2**: Model calibration and simulation of the capital per worker path.
  - **Part 3**: Simulated steady-state output comparison with actual GDP data.

### 📊 Data Files
- **`India_Data.csv`**: Dataset prepared from FRED, containing GDP, capital stock, labor force, and other relevant economic indicators.
- **`[india_growth_processed.csv]`**: This file has the solow residual calculated to use for Part 2 and 3.

### 📄 Supplementary Files
- **Part 4 Text Files**: Contain problem setup and numerical solutions for the intertemporal optimization problem.
- **`FOC_calc.pdf`**: Contains manual calculations for Part 4 and detailed project guidelines.

---

## Running the Code

### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install numpy pandas matplotlib
```

### Execution Steps
1. **Clone the repository** (or download the files).
2. **Navigate to the directory** where the files are located.
3. **Run `prefinal.py`** to perform Parts 1, 2, and 3:
   ```bash
   python prefinal.py
   ```
4. **For Part 4**, refer to the text files and PDF for manual and numerical solutions.

---

##  Results and Output
- The script will generate tables and visualizations to analyze economic growth trends and model accuracy.
- Part 4 results can be compared manually and numerically based on sensitivity analysis.

---

##  Notes
- The **base year for calibration** should be chosen carefully based on data availability.
- Adjust parameters (**savings rate, depreciation rate, capital elasticity**) as needed to test different model scenarios.








