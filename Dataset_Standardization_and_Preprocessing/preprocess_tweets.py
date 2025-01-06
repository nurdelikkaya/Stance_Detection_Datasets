import pandas as pd
import os

# Directory containing the CSV files
directory = 'datasets/table3'

# Name of the output file to skip
output_filename = 'combined_cleaned_table3.csv'

# Initialize an empty list to hold DataFrames
dataframes = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv') and filename != output_filename:  # Process only CSV files, excluding the output file
        file_path = os.path.join(directory, filename)
        try:
            # Load the file with error handling
            data = pd.read_csv(
                file_path,
                delimiter='\t',
                engine="python",
                quoting=3,  # Prevents pandas from interpreting quotes
                on_bad_lines='skip'  # Skips problematic lines
            )

            # Drop rows with missing 'Text' or 'Class' values
            data = data.dropna(subset=['Text', 'Class'])

            # Append the cleaned DataFrame to the list
            dataframes.append(data)
            print(f"Processed file: {filename}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Concatenate all the DataFrames
combined_data = pd.concat(dataframes, ignore_index=True)

# Save the combined dataset to a new CSV file
combined_file_path = os.path.join(directory, output_filename)
combined_data.to_csv(combined_file_path, index=False)
print(f"Combined dataset saved to {combined_file_path}")
