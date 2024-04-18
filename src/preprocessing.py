import pandas as pd
import numpy as np

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

    return desc, global_radio, multislice

def complete_df(global_radio, multislice):

    global_radio_complete = global_radio[global_radio.groupby('ID')['temps_inj'].transform('nunique') == 4]
    multislice_complete = multislice[multislice['ID'].isin(global_radio_complete['ID'])]

    return global_radio_complete, multislice_complete

def prepare_feature_target_matrices(df, target_col_prefix='classe_name'):

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



    