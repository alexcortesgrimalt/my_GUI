import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 1. Path configuration
path = './*.csv' 
files = glob.glob(path)

if not files:
    print("No .csv files found in the folder.")

# List to store data for sorted plotting
plot_data = []

for file in files:
    try:
        filename = os.path.basename(file)
        # Extract numeric value for 'd'
        d_str = filename.replace('.csv', '')
        d_value = float(d_str)

        # Automatic detection of data start
        skip = 0
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if 'I1,V1' in line:
                    skip = i + 2 
                    break
        
        # Read CSV and clean
        df = pd.read_csv(file, skiprows=skip, header=None, usecols=[0, 1], names=['I', 'V'])
        df = df.dropna()
        
        # Store tuple (d_value, dataframe)
        plot_data.append((d_value, df))
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 2. Sort data from largest to smallest based on d_value
plot_data.sort(key=lambda x: x[0], reverse=True)

# 3. Plotting
plt.figure(figsize=(10, 6))

for d, df in plot_data:
    plt.plot(df['V'], df['I'], label=f'{d} µm')

# 4. Aesthetics (English)
plt.title('I-V Characteristic Curves')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.legend(title="Antenna Gap (d)", loc='best')
plt.tight_layout()

plt.show()