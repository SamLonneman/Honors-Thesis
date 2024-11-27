import pandas as pd
import numpy as np

def calculate_velocities_and_deviations(input_file, output_file):
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Calculate time difference in seconds
    df['TimeDiff'] = df['UnixTime'].diff() / 1000  # Assuming UnixTime is in milliseconds
    
    # Calculate hand velocities (right hand)
    for axis in ['X', 'Y', 'Z']:
        # HoloLens right hand velocities
        df[f'HoloRightHandVel{axis}'] = df[f'HoloRightHandPos{axis}'].diff() / df['TimeDiff']
        # Optitrack right hand velocities
        df[f'OptitrackRightHandVel{axis}'] = df[f'OptitrackRightHandPos{axis}'].diff() / df['TimeDiff']
    
    # Calculate head velocities
    for axis in ['X', 'Y', 'Z']:
        df[f'HeadVel{axis}'] = df[f'Head{axis}'].diff() / df['TimeDiff']
    
    # Calculate net velocities (magnitude)
    def calculate_net_velocity(vx, vy, vz):
        return np.sqrt(vx**2 + vy**2 + vz**2)
    
    df['HoloRightHandNetVel'] = calculate_net_velocity(
        df['HoloRightHandVelX'], df['HoloRightHandVelY'], df['HoloRightHandVelZ'])
    
    df['OptitrackRightHandNetVel'] = calculate_net_velocity(
        df['OptitrackRightHandVelX'], df['OptitrackRightHandVelY'], df['OptitrackRightHandVelZ'])
    
    df['HeadNetVel'] = calculate_net_velocity(
        df['HeadVelX'], df['HeadVelY'], df['HeadVelZ'])
    
    # Calculate average positions
    holo_avg_pos = df[['HoloRightHandPosX', 'HoloRightHandPosY', 'HoloRightHandPosZ']].mean()
    opti_avg_pos = df[['OptitrackRightHandPosX', 'OptitrackRightHandPosY', 'OptitrackRightHandPosZ']].mean()
    head_avg_pos = df[['HeadX', 'HeadY', 'HeadZ']].mean()
    
    # Calculate deviations from average positions
    for axis in ['X', 'Y', 'Z']:
        df[f'HoloRightHandDeviation{axis}'] = df[f'HoloRightHandPos{axis}'] - holo_avg_pos[f'HoloRightHandPos{axis}']
        df[f'OptitrackRightHandDeviation{axis}'] = df[f'OptitrackRightHandPos{axis}'] - opti_avg_pos[f'OptitrackRightHandPos{axis}']
        df[f'HeadDeviation{axis}'] = df[f'Head{axis}'] - head_avg_pos[f'Head{axis}']
    
    # Calculate net deviations
    df['HoloRightHandNetDeviation'] = calculate_net_velocity(
        df['HoloRightHandDeviationX'], df['HoloRightHandDeviationY'], df['HoloRightHandDeviationZ'])
    
    df['OptitrackRightHandNetDeviation'] = calculate_net_velocity(
        df['OptitrackRightHandDeviationX'], df['OptitrackRightHandDeviationY'], df['OptitrackRightHandDeviationZ'])
    
    df['HeadNetDeviation'] = calculate_net_velocity(
        df['HeadDeviationX'], df['HeadDeviationY'], df['HeadDeviationZ'])
    
    # Save updated DataFrame
    df.to_csv(output_file, index=False)

# Usage
if __name__ == "__main__":

    input_file = 'P48_Log_With_Codes.csv'
    output_file = 'P48_Log_With_Velocities.csv'
    calculate_velocities_and_deviations(input_file, output_file)
