import pandas as pd
import numpy as np

def add_features(csv_file, output_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure required columns exist for right hand
    required_columns_right = ['HoloRightHandPosX', 'HoloRightHandPosY', 'HoloRightHandPosZ']
    for col in required_columns_right:
        if col not in data.columns:
            raise ValueError(f"The CSV file must contain a '{col}' column.")
    
    # Ensure required columns exist for left hand
    required_columns_left = ['HoloLeftHandPosX', 'HoloLeftHandPosY', 'HoloLeftHandPosZ']
    for col in required_columns_left:
        if col not in data.columns:
            raise ValueError(f"The CSV file must contain a '{col}' column.")
    
    # Ensure required columns exist for head
    required_columns_head = ['HeadX', 'HeadY', 'HeadZ']
    for col in required_columns_head:
        if col not in data.columns:
            raise ValueError(f"The CSV file must contain a '{col}' column.")
    
    # Calculate the average position for right hand
    avg_pos_right = data[required_columns_right].mean()
    
    # Calculate the average position for left hand
    avg_pos_left = data[required_columns_left].mean()
    
    # Calculate the average position for head
    avg_pos_head = data[required_columns_head].mean()
    
    # Calculate the displacement from the average position for right hand
    data['HoloRightHandDistance'] = np.sqrt(
        (data['HoloRightHandPosX'] - avg_pos_right['HoloRightHandPosX'])**2 +
        (data['HoloRightHandPosY'] - avg_pos_right['HoloRightHandPosY'])**2 +
        (data['HoloRightHandPosZ'] - avg_pos_right['HoloRightHandPosZ'])**2
    )
    
    # Calculate the displacement from the average position for left hand
    data['HoloLeftHandDistance'] = np.sqrt(
        (data['HoloLeftHandPosX'] - avg_pos_left['HoloLeftHandPosX'])**2 +
        (data['HoloLeftHandPosY'] - avg_pos_left['HoloLeftHandPosY'])**2 +
        (data['HoloLeftHandPosZ'] - avg_pos_left['HoloLeftHandPosZ'])**2
    )
    
    # Calculate the displacement from the average position for head
    data['HeadDistance'] = np.sqrt(
        (data['HeadX'] - avg_pos_head['HeadX'])**2 +
        (data['HeadY'] - avg_pos_head['HeadY'])**2 +
        (data['HeadZ'] - avg_pos_head['HeadZ'])**2
    )
    
    # Calculate the speed for right hand (assuming the data is time-ordered and sampled at a constant rate)
    data['HoloRightHandSpeed'] = np.sqrt(
        data['HoloRightHandPosX'].diff()**2 +
        data['HoloRightHandPosY'].diff()**2 +
        data['HoloRightHandPosZ'].diff()**2
    )
    
    # Calculate the speed for left hand (assuming the data is time-ordered and sampled at a constant rate)
    data['HoloLeftHandSpeed'] = np.sqrt(
        data['HoloLeftHandPosX'].diff()**2 +
        data['HoloLeftHandPosY'].diff()**2 +
        data['HoloLeftHandPosZ'].diff()**2
    )
    
    # Calculate the speed for head (assuming the data is time-ordered and sampled at a constant rate)
    data['HeadSpeed'] = np.sqrt(
        data['HeadX'].diff()**2 +
        data['HeadY'].diff()**2 +
        data['HeadZ'].diff()**2
    )
    
    # Calculate the distance between right and left hands
    data['DistanceBetweenHands'] = np.sqrt(
        (data['HoloRightHandPosX'] - data['HoloLeftHandPosX'])**2 +
        (data['HoloRightHandPosY'] - data['HoloLeftHandPosY'])**2 +
        (data['HoloRightHandPosZ'] - data['HoloLeftHandPosZ'])**2
    )
    
    # Save the updated dataset to a new CSV file
    data.to_csv(output_file, index=False)

# Example usage
csv_file = 'P48_Log_NEW_With_Codes.csv'
output_file = 'P48_Log_NEW_With_Codes_with_features.csv'
add_features(csv_file, output_file)
