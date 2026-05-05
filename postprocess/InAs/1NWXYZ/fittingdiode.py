import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- Constants ---
q = 1.602e-19
kB = 1.38e-23
T = 300
Vt = (kB * T) / q

def schottky_model(V, Is_scaled, n):
    """Schottky model with Is in nA for numerical stability"""
    return (Is_scaled * 1e-9) * (np.exp(np.clip(V / (n * Vt), -700, 700)) - 1)

def analyze_iv(file_path):
    # 1. Parsing the Keithley CSV
    skip = 0
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if 'I1,V1' in line:
                    skip = i + 2
                    break
        
        df = pd.read_csv(file_path, skiprows=skip, header=None, usecols=[0, 1], names=['I', 'V'])
        df = df.dropna()
        
        # Split into forward and absolute for log analysis
        forward_mask = (df['V'] > 0.1) & (df['I'] > 0)
        v_fwd = df.loc[forward_mask, 'V'].values
        i_fwd = df.loc[forward_mask, 'I'].values

        # 2. Fitting Schottky (Reference)
        try:
            popt, _ = curve_fit(schottky_model, v_fwd, i_fwd, p0=[0.001, 2.0], 
                               bounds=((1e-12, 1.0), (1000, 60.0)))
            is_fit = popt[0] * 1e-9
            n_fit = popt[1]
        except:
            is_fit, n_fit = None, None

        # 3. Power Law Analysis (Slope m in Log-Log)
        # log(I) = m * log(V) + C -> I = V^m
        log_v = np.log10(v_fwd)
        log_i = np.log10(i_fwd)
        m, intercept = np.polyfit(log_v, log_i, 1)

        # 4. Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Subplot 1: Linear Scale + Schottky Fit ---
        ax1.scatter(df['V'], df['I'] * 1e12, color='black', s=10, label='Data')
        if is_fit:
            v_plot = np.linspace(0, max(v_fwd), 100)
            i_plot = schottky_model(v_plot, *popt)
            ax1.plot(v_plot, i_plot * 1e12, 'r--', label=f'Schottky Fit (n={n_fit:.2f})')
        
        ax1.set_title('Linear Scale (pA)')
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (pA)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Subplot 2: Log-Log Scale (Power Law Analysis) ---
        ax2.loglog(v_fwd, i_fwd, 'bo', markersize=4, label='Data')
        ax2.plot(v_fwd, 10**(m * np.log10(v_fwd) + intercept), 'g-', 
                 label=f'Slope m = {m:.2f}')
        
        ax2.set_title('Log-Log Analysis ($I \propto V^m$)')
        ax2.set_xlabel('log Voltage (V)')
        ax2.set_ylabel('log Current (A)')
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)

        print(f"--- Analysis for {os.path.basename(file_path)} ---")
        print(f"Power Law Slope (m): {m:.2f}")
        if is_fit:
            print(f"Schottky Is: {is_fit:.4e} A")
            print(f"Schottky n: {n_fit:.2f}")
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not process file: {e}")

# Run the analysis
analyze_iv('32.csv')