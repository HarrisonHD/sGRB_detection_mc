import re
import pandas as pd


# Function to extract numeric values from columns and keep the first number
def clean_numeric_column(df, column_name):
    # Extract the first number from each entry
    df[column_name] = df[column_name].astype(str).apply(
        lambda x: re.findall(r'[\d.]+', x)[0] if re.findall(r'[\d.]+', x) else None)
    return df


# Function to clean the redshift column designed to select certain values depending on what is given
def clean_numeric_column_z(df, column_name):
    def extract_value(text):
        # Patterns for each category
        patterns = {
            'emission': r'emission.*?([\d.]+)',
            'absorption': r'absorption.*?([\d.]+)',
            'photometry': r'photometry.*?([\d.]+)'
        }

        # Search for each category and extract the first matching numeric value
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        # If no category-specific value is found, extract the number with the most decimal points
        all_numbers = re.findall(r'[\d.]+', text)
        if all_numbers:
            return max(all_numbers, key=lambda x: (len(x.split('.')[-1]) if '.' in x else 0, x))
        return None

    # Apply the extraction function to the specified column
    df[column_name] = df[column_name].astype(str).apply(extract_value)
    return df


# Function to convert columns to numeric values
def convert_columns_to_numeric(df, columns_to_exclude):
    for col in df.columns:
        if col not in columns_to_exclude:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Function to calculate T_intrinsic
def calculate_T_intrinsic(df, duration_col, redshift_col):
    df['T_intrinsic'] = df[duration_col] / (1 + df[redshift_col])
    return df


def split_and_update_photon_index(df):
    def split_photon_index(row):
        # Convert row to string and split by comma
        parts = str(row).split(',')

        # Clean up the number and text parts
        number_part = parts[0].strip() if len(parts) > 0 else ''
        text_part = parts[1].strip() if len(parts) > 1 else ''

        return number_part, text_part

    # split the column and add the results as new columns
    df[['BAT Photon Index Value', 'BAT Photon Index Type']] = df[
        'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].apply(
        split_photon_index).apply(pd.Series)

    # Drop the original column
    df.drop('BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)', axis=1, inplace=True)
    return df


# Load data
grb_table_rf = pd.read_csv('Data/grb_table_rf.txt', sep='\t')
grb_table = pd.read_csv('Data/grb_table.txt', sep='\t')

# Clean and convert 'Redshift' column in grb_table_rf
grb_table_rf = clean_numeric_column_z(grb_table_rf, 'Redshift')
grb_table = clean_numeric_column_z(grb_table, 'Redshift')

# Split the photon index column into two columns
grb_table_rf = split_and_update_photon_index(grb_table_rf)
grb_table = split_and_update_photon_index(grb_table)

# List of columns to keep as strings
columns_to_exclude = [
    'GRB',
    'BAT Photon Index Type',
    'XRT RA (J2000)',
    'XRT Dec (J2000)'
]

# Convert the columns to numeric values
grb_table_rf = convert_columns_to_numeric(grb_table_rf, columns_to_exclude)
grb_table = convert_columns_to_numeric(grb_table, columns_to_exclude)

# Calculate T_intrinsic for grb_table_rf and grb_table
grb_table_rf = calculate_T_intrinsic(grb_table_rf, 'BAT T90 [sec]', 'Redshift')
grb_table = calculate_T_intrinsic(grb_table, 'BAT T90 [sec]', 'Redshift')

# Filter both sets of data for T_intrinsic times of 2 seconds or less
filtered_grb_table_rf = grb_table_rf[grb_table_rf['T_intrinsic'] <= 2]
filtered_grb_table = grb_table[grb_table['T_intrinsic'] <= 2]

# Separate sGRBs without redshifts and export them to a separate file
sgrbs_without_redshift = grb_table[grb_table['Redshift'].isna()]

# Merge the data
combined_grb_table = pd.concat([filtered_grb_table_rf, filtered_grb_table], ignore_index=True)

# Removing duplicates if there are any
combined_grb_table = combined_grb_table.drop_duplicates()

# Exporting the combined dataset
output_file_path = 'Data/combined_grb_table.txt'
sgrbs_without_redshift.to_csv(output_file_path, sep='\t', index=False)
combined_grb_table.to_csv(output_file_path, sep='\t', index=False)
print(f"Combined GRB data exported to {output_file_path}")
print(f"Total Number of sGRB: {len(combined_grb_table)}")
