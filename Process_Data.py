import pandas as pd

def time_to_milliseconds(time_str, offset_ms=0):

    # Split into time and tenths
    time_part, tenths = time_str.split(',')
    hours, minutes, seconds = map(int, time_part.split(':'))
    
    # Calculate total milliseconds
    total_ms = (hours * 3600000 +
                minutes * 60000 +
                seconds * 1000 +
                int(tenths) * 100)
    
    return total_ms - offset_ms

def process_csv(input_file, output_file, offset_time="00:00:00,0"):

    # Calculate offset in ms
    offset_ms = time_to_milliseconds(offset_time, 0)
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Convert beginning and end columns to milliseconds, taking into account the sync offset
    df['Beginning_Time'] = df['Beginning'].apply(lambda x: time_to_milliseconds(x, offset_ms))
    df['Ending_Time'] = df['End'].apply(lambda x: time_to_milliseconds(x, offset_ms))
    
    # Select relevant columns and save to output file
    df = df[['Beginning_Time', 'Ending_Time', 'Code']]
    df.to_csv(output_file, index=False)

def add_codes_to_log(log_file, segments_file, output_file, offset_ms=0):

    # Read files (already sorted by time)
    log_df = pd.read_csv(log_file)
    segments_df = pd.read_csv(segments_file)
    
    # Initialize Code column
    log_df['Code'] = None
    
    current_log_index = 0
    log_length = len(log_df)
    
    # Single pass through non-overlapping segments
    for _, segment in segments_df.iterrows():
        segment_start = segment['Beginning_Time']
        segment_end = segment['Ending_Time']
        
        # Advance to segment start
        while current_log_index < log_length and (log_df.iloc[current_log_index]['AppTime'] - offset_ms) < segment_start:
            if current_log_index % 100 == 0:
                print(f"Applying codes to log file... Progress: {current_log_index / log_length * 100:.2f}%", end='\r')
            current_log_index += 1
        
        # Mark all entries in this segment
        while current_log_index < log_length and (log_df.iloc[current_log_index]['AppTime'] - offset_ms) <= segment_end:
            if current_log_index % 100 == 0:
                print(f"Applying codes to log file... Progress: {current_log_index / log_length * 100:.2f}%", end='\r')
            log_df.iloc[current_log_index, log_df.columns.get_loc('Code')] = segment['Code']
            current_log_index += 1
    
    # Save to output file
    print('Applying codes to log file... Progress: 100.00%')
    save_with_progress(log_df, output_file)

def save_with_progress(df, output_file, chunk_size=1000):
    total_rows = len(df)
    for i in range(0, total_rows, chunk_size):
        chunk = df[i:i + chunk_size]
        if i == 0:
            chunk.to_csv(output_file, index=False, mode='w')
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)
        progress = min((i + chunk_size) / total_rows * 100, 100)
        print(f"Saving processed log file...  Progress: {progress:.2f}%", end='\r')
    print('Saving processed log file...  Progress: 100.00%')

def process_all(raw_segments_file, raw_log_file, offset_time, offset_ms):

    # Define intermediate and output files
    processed_segments_file = raw_segments_file.replace(".csv", "_processed.csv")
    final_log_file = raw_log_file.replace(".csv", "_With_Codes.csv")
    
    try:
        process_csv(raw_segments_file, processed_segments_file, offset_time)
        add_codes_to_log(raw_log_file, processed_segments_file, final_log_file, offset_ms)
        print(f"Processing complete. Output saved to \"{final_log_file}\"")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":

    raw_segments = "P48_Coded_Segments.csv" # Coded segments file
    raw_log = "P48_Log.csv"                 # Log file
    sync_segments = "0:28:44,6"             # Sync time for segments
    sync_logs = 2028992                     # Sync time for logs
    
    process_all(raw_segments, raw_log, sync_segments, sync_logs)