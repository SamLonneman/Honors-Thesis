import pandas as pd
import numpy as np

def calculate_normalized_averages(input_file, output_file):
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Get numeric columns only, excluding 'Code'
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns[numeric_columns != 'Code']
    
    # Normalize numeric columns using min-max scaling
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0  # Set to 0 if all values are the same
    
    # Calculate means for normalized columns grouped by Code
    averages = df.groupby('Code')[numeric_columns].mean()
    
    # Add count of samples per code
    averages['SampleCount'] = df.groupby('Code').size()
    
    # Save results
    averages.to_csv(output_file)
    
    # Print summary
    print(f"Normalized averages calculated and saved to {output_file}")
    print("\nSample counts per code:")
    print(averages['SampleCount'])

if __name__ == "__main__":
    input_file = "P48_Log_With_Velocities.csv"
    output_file = "P48_Log_Averages_Normalized.csv"
    calculate_normalized_averages(input_file, output_file)