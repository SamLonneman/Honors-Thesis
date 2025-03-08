import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_boxplots(csv_file, columns, output_folder='BoxPlots'):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure 'Code' column exists
    if 'Code' not in data.columns:
        raise ValueError("The CSV file must contain a 'Code' column.")
    
    # Filter out rows where speed is 0 (these are moments that objects are lost)
    data = data[data['HoloRightHandSpeed'] != 0]
    data = data[data['HoloLeftHandSpeed'] != 0]
    data = data[data['HeadSpeed'] != 0]
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate boxplots for each specified column
    for column in columns:
        if column in data.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Code', y=column, data=data, showfliers=False)
            plt.title(f'Boxplot of {column} by Code')
            plt.xlabel('Code')
            plt.ylabel(column)
            output_path = os.path.join(output_folder, f'{column}_boxplot.png')
            plt.savefig(output_path)
            plt.close()
        else:
            print(f"Column '{column}' not found in the CSV file.")

# Example usage
csv_file = 'P48_Log_NEW_With_Codes_with_features.csv'
columns_to_plot = ['HoloRightHandDistance', 'HoloRightHandSpeed', 'HoloLeftHandDistance', 'HoloLeftHandSpeed', 'DistanceBetweenHands', 'HeadDistance', 'HeadSpeed']

generate_boxplots(csv_file, columns_to_plot)
