import pandas

def join_csv_files(file_list, output_file):
    """Join multiple CSV files into a single CSV file.
    
    Args:
        file_list: List of file paths to CSV files to be joined.
        output_file: Path to the output CSV file.
    """
    dataframes = []
    
    for file in file_list:
        df = pandas.read_csv(file)
        # Keep only fullText and id columns if they exist
        available_cols = df.columns.tolist()
        cols_to_keep = [col for col in ['id', 'fullText'] if col in available_cols]
        
        if cols_to_keep:
            df = df[cols_to_keep]
            dataframes.append(df)
            print(f"Read {file}: {len(df)} rows, columns: {cols_to_keep}")
        else:
            print(f"Warning: {file} does not contain 'id' or 'fullText' columns")
    
    combined_df = pandas.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Joined {len(file_list)} files into {output_file}")

if __name__ == "__main__":
    # Files are inside Data folder
    # take all files that end with .csv
    import os
    data_folder = "Data"
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]
    output_csv = os.path.join(data_folder, "combined_data.csv")
    join_csv_files(csv_files, output_csv)