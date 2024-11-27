import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_metric_comparison_plots(csv_file):
    # Create Graphs directory
    graphs_dir = "Graphs_Normalized"
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file, index_col=0)
    
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("husl")
    
    # Define metrics categories
    categories = {
        'Position': {
            'Head': ['HeadX', 'HeadY', 'HeadZ'],
            'Gaze': ['GazeX', 'GazeY', 'GazeZ'],
            'HoloRightHand': ['HoloRightHandPosX', 'HoloRightHandPosY', 'HoloRightHandPosZ']
        },
        'Velocity': {
            'HandVelocity': ['HoloRightHandVelX', 'HoloRightHandVelY', 'HoloRightHandVelZ'],
            'NetVelocity': ['HeadNetVel', 'HoloRightHandNetVel']
        },
        'Physiological': {
            'Vitals': ['Temperature', 'GSR', 'Hr', 'BVP']
        },
        'Deviation': {
            'NetDeviation': ['HoloRightHandNetDeviation', 'HeadNetDeviation']
        }
    }
    
    for main_category, subcategories in categories.items():
        for subcat_name, metrics in subcategories.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Extract data for plotting
            plot_data = df[metrics]
            
            # Create grouped bar plot
            x = np.arange(len(metrics))
            width = 0.1
            n_codes = len(df.index)
            
            for i, code in enumerate(df.index):
                offset = width * (i - n_codes/2)
                ax.bar(x + offset, df.loc[code, metrics], width, label=code)
            
            # Customize plot
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.set_title(f'{subcat_name} Comparison Across Activities')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            save_path = os.path.join(graphs_dir, f'{main_category}_{subcat_name}_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create sample count comparison
    plt.figure(figsize=(10, 6))
    df['SampleCount'].plot(kind='bar')
    plt.title('Sample Count by Activity')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    save_path = os.path.join(graphs_dir, 'sample_count_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_metric_comparison_plots('P48_Log_Averages_Normalized.csv')