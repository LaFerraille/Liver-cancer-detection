import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_df(desc, global_radio, multislice):

    # Copy the dataframes
    desc = desc.copy()
    global_radio = global_radio.copy()
    multislice = multislice.copy()

    # Create "ID_temp" column in desc, global_radio and multislice based on classe_name x patient_num
    desc['ID_temp'] = desc['classe_name'] + desc['patient_num'].astype(str)
    desc['ID'] = desc.index

    # Now create "ID" column in global_radio and multislice based on ID_temp
    global_radio['ID_temp'] = global_radio['classe_name'] + global_radio['patient_num'].astype(str)
    multislice['ID_temp'] = multislice['classe_name'] + multislice['patient_num'].astype(str)

    # Add "ID" column to global_radio and multislice
    global_radio = pd.merge(global_radio, desc[['ID', 'ID_temp']], on='ID_temp', how='left')
    multislice = pd.merge(multislice, desc[['ID', 'ID_temp']], on='ID_temp', how='left')

    # Drop "ID_temp" column from all dataframes
    desc.drop(columns=['ID_temp'], inplace=True)
    global_radio.drop(columns=['ID_temp'], inplace=True)
    multislice.drop(columns=['ID_temp'], inplace=True)

    # Drop column "patient_num" in desc, global_radio and multislice
    desc.drop(columns=['patient_num'], inplace=True)
    global_radio.drop(columns=['patient_num'], inplace=True)
    multislice.drop(columns=['patient_num'], inplace=True)

    # Move column "ID" to the first position in desc, global_radio and multislice
    desc = desc[['ID'] + [col for col in desc.columns if col != 'ID']]
    global_radio = global_radio[['ID'] + [col for col in global_radio.columns if col != 'ID']]
    multislice = multislice[['ID'] + [col for col in multislice.columns if col != 'ID']]

    # Sort desc, global_radio and multislice by "ID" and "temps_inj" as ART > PORT > VEIN > TARD
    inj_order = ['ART', 'PORT', 'VEIN', 'TARD']
    order_mapping = {inj: i for i, inj in enumerate(inj_order)}

    global_radio['temps_inj'] = pd.Categorical(global_radio['temps_inj'], categories=inj_order, ordered=True)
    multislice['temps_inj'] = pd.Categorical(multislice['temps_inj'], categories=inj_order, ordered=True)

    desc = desc.sort_values(by='ID').reset_index(drop=True)
    global_radio = global_radio.sort_values(by=['ID', 'temps_inj']).reset_index(drop=True)
    multislice = multislice.sort_values(by=['ID', 'temps_inj']).reset_index(drop=True)

    # Ge rid of the columns starting by "diagnostics_"
    global_radio = global_radio.loc[:, ~global_radio.columns.str.startswith('diagnostics_')]
    multislice = multislice.loc[:, ~multislice.columns.str.startswith('diagnostics_')]

    # Drop "Column1" column from global_radio and multislice
    global_radio.drop(columns=['Column1'], inplace=True)
    multislice.drop(columns=['Column1'], inplace=True)

    return desc, global_radio, multislice

# Pas nécessaire avec PARAFAC
def complete_df(global_radio, multislice):
    # Filter global_radio to include only IDs that have exactly 4 unique temps_inj entries
    global_radio_complete = global_radio[global_radio.groupby('ID')['temps_inj'].transform('nunique') == 4]

    # Filter multislice to include only IDs from global_radio_complete
    multislice = multislice[multislice['ID'].isin(global_radio_complete['ID'])]

    # Group by 'ID' and 'temps_inj' in multislice and count the number of slices
    slice_counts = multislice.groupby(['ID', 'temps_inj']).size().reset_index(name='slice_count')

    # Pivot the table to have temps_inj as columns and slice_count as values
    pivot_table = slice_counts.pivot(index='ID', columns='temps_inj', values='slice_count')

    # Ensure we're only considering IDs where data is available for all four temps_inj
    # Assuming these are 'ART', 'VEN', 'DELAY', 'LATE'
    pivot_table = pivot_table.dropna(subset=['ART', 'VEIN', 'PORT', 'TARD'])

    # Find IDs where slice_count is the same across all four temps_inj
    consistent_ids = pivot_table[pivot_table.std(axis=1) == 0].index

    # Filter both DataFrames to only include consistent IDs
    multislice_complete = multislice[multislice['ID'].isin(consistent_ids)]

    # Convert slice_num to integer
    multislice_complete['slice_num'] = multislice_complete['slice_num'].astype(int)

    return global_radio_complete, multislice_complete


def multislice_consistent(multislice_complete, min_range=10):
    # Get unique IDs from the DataFrame
    unique_ids = multislice_complete['ID'].unique()

    # Dictionary to store the range of each ID
    id_ranges = {}

    for id in unique_ids:
        patient_data = multislice_complete[multislice_complete['ID'] == id]
        min_slice = patient_data['slice_num'].min()
        max_slice = patient_data['slice_num'].max()
        slice_range = max_slice - min_slice
        id_ranges[id] = slice_range

    # List IDs with slice range less than the specified minimum range
    ids_to_keep = [id for id, range in id_ranges.items() if range >= min_range]

    # Filter DataFrame to only include IDs with sufficient slice range
    filtered_df = multislice_complete[multislice_complete['ID'].isin(ids_to_keep)]

    # Further refine the DataFrame to include only the middle 10 slices per ID and temps_inj
    middle_slices_df = pd.DataFrame()  # Initialize an empty DataFrame to hold results

    for (id, temps_inj), group in filtered_df.groupby(['ID', 'temps_inj']):
        # Calculate the middle range
        middle_index = (group['slice_num'].max() + group['slice_num'].min()) // 2
        slice_start = max(middle_index - 5, group['slice_num'].min())
        slice_end = min(middle_index + 4, group['slice_num'].max())

        # Filter to get only the middle 10 slices (or as many as possible if less than 10)
        middle_slices = group[(group['slice_num'] >= slice_start) & (group['slice_num'] <= slice_end)]
        
        # Append to the result DataFrame
        middle_slices_df = pd.concat([middle_slices_df, middle_slices])

    return middle_slices_df


def X_y_global(df):

    df.set_index(['ID', 'temps_inj'], inplace=True)

    # Unstack 'temps_inj' to create multi-level column names
    wide_df = df.unstack(level='temps_inj')

    # Create new column names combining feature names and temps_inj labels
    new_columns = [f"{multi_idx}_{col}" for col, multi_idx in wide_df.columns]
    wide_df.columns = new_columns

    # Separate the feature columns and target column
    X = wide_df.drop([col for col in new_columns if 'classe_name' in col], axis=1)
    y = wide_df['ART_classe_name']  # Assuming classe_name is the same across all 'temps_inj'

    return X, y

def X_y_multislice(df):

    # We first normalize the slice_num across all patients and temps_inj
    current_id = None
    current_temps_inj = None
    slice_counter = 1

    for index, row in df.iterrows():
        # Check if we are still on the same ID and temps_inj
        if (row['ID'], row['temps_inj']) == (current_id, current_temps_inj):
            # Continue incrementing the slice counter
            slice_counter += 1
        else:
            # Reset for a new ID or temps_inj
            slice_counter = 1
            current_id = row['ID']
            current_temps_inj = row['temps_inj']

        # Set the slice number from 1 to 10
        df.at[index, 'slice_num'] = slice_counter

    df.set_index(['ID', 'temps_inj', 'slice_num'], inplace=True)

    # Unstack 'temps_inj' to create multi-level column names, keep 'slice_num' in the index
    wide_df = df.unstack(level=['temps_inj', 'slice_num'])

    new_columns = [f"{temps_inj}_{slice_num}_{col}" for (col, temps_inj, slice_num) in wide_df.columns]
    wide_df.columns = new_columns

    classe_columns = [col for col in new_columns if 'classe_name' in col]

    X = wide_df.drop(classe_columns, axis=1)
    y = wide_df['ART_1_classe_name']  # Use this column as the target if it exists

    #Remove line from X where there is more than 300 NaN and the corresponding line in y
    threshold = X.shape[1] - 300
    X = X.dropna(thresh=threshold, axis=0)
    y = y.loc[X.index]

    return X, y

def rgcca_df(X, y):

    y_encoded = pd.DataFrame({
        'CCK': (y == 'CCK').astype(int),
        'CHC': (y == 'CHC').astype(int),
        'Mixtes': (y == 'Mixtes').astype(int)
    })
    
    # Standardizing the feature matrix 'vein_features'
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_multislice_rgcca = pd.concat([df_scaled, y_encoded], axis=1)
    X_multislice_rgcca.to_csv('data/processed/multislice_rgcca.csv', index=False)

    return X_multislice_rgcca




    