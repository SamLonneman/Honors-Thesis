import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_correlation_plots(csv_file):
    # Create Graphs directory
    graphs_dir = "Graphs_Correlation"
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Read data
    df = pd.read_csv(csv_file, index_col=0)
    
    # Create correlation matrix heatmap without annotations
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of All Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create box plots for key metrics
    key_metrics = ['HeadNetVel', 'HoloRightHandNetVel', 'Temperature', 'GSR', 'Hr']
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=df.index, y=metric)
        plt.title(f'{metric} Distribution by Code')
        plt.xticks(rotation=45)
        plt.tight_layout()

if __name__ == "__main__":
    create_correlation_plots('P48_Log_Averages.csv')