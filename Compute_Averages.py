import pandas as pd
import numpy as np

def calculate_averages_by_code(input_file, output_file):
    # Read the processed data
    df = pd.read_csv(input_file)
    
    # Get numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate means for numeric columns grouped by Code
    averages = df.groupby('Code')[numeric_columns].mean()
    
    # Add count of samples per code
    averages['SampleCount'] = df.groupby('Code').size()
    
    # Save results
    averages.to_csv(output_file)
    
    # Print summary
    print(f"Averages calculated and saved to {output_file}")
    print("\nSample counts per code:")
    print(averages['SampleCount'])

if __name__ == "__main__":
    input_file = "P48_Log_With_Velocities.csv"
    output_file = "P48_Log_Averages.csv"
    calculate_averages_by_code(input_file, output_file)
