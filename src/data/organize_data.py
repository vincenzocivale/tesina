import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import SMOTE
from sklearn.preprocessing import StandardScaler

def organize_data2(input_filepath, features_list):
    """
    Organize the data into a format that is ready for analysis
    """

    df = pd.DataFrame(columns=['PatientID', 'SignalID', 'Group'] + features_list)

    for root, dirs, files in os.walk(input_filepath):
        for file in files:
            if file.endswith('.csv'):
                signal_id = file.split('#')[-1].split('.')[0]

                # sottogruppo del paziente
                relative_path = os.path.relpath(root, input_filepath)
                patient_group = relative_path.split(os.sep)[0]
                
                if patient_group=='mentalDisorders_MIMIC_125' or patient_group=='sepsis_MIMIC_125':
                    patient_id = file.split('-')[0].split('.')[0]
                    
                    signal_id = file.split('-')[1].split('.')[0]
                
                elif patient_group=='covid_Empoli_60' or patient_group=='healthyControl_Empoli_60':
                    patient_id = file.split('_')[0]

                    signal_id = file.split('_')[1].split('.')[0]

                full_file_path = os.path.join(root, file)  # Percorso completo del file

                data_df = pd.read_csv(full_file_path)

                data_df.drop(['start_time', 'end_time'], axis=1, inplace=True)

                # Aggiungi le nuove colonne con valori costanti alle prime tre posizioni
                data_df.insert(0, 'PatientID', patient_id)
                data_df.insert(1, 'SignalID', signal_id)
                data_df.insert(2, 'Group', patient_group)

                df = pd.concat([df, data_df], ignore_index=True)
            
    return df



def salva_dati_per_gruppo(df, percorso_base):
    """
    Salva i dati del dataframe in file .npz organizzati per gruppi.
    
    df: DataFrame con le colonne 'Group', 'PatientID', 'SignalID', e 'Data'.
    percorso_base: Path di base per la creazione delle cartelle.
    """
    
    if not os.path.exists(percorso_base):
        os.makedirs(percorso_base)
    
    gruppi = df.groupby('Group')
    
    for nome_gruppo, gruppo in gruppi:
        
        percorso_gruppo = os.path.join(percorso_base, str(nome_gruppo))
        
        if not os.path.exists(percorso_gruppo):
            os.makedirs(percorso_gruppo)
        
        for idx, riga in gruppo.iterrows():
            patient_id = riga['PatientID']
            signal_id = riga['SignalID']
            data = np.array(riga['Data'])
            
            min_val = data.min()
            max_val = data.max()
            data_normalized = (data - min_val) / (max_val - min_val)  # Min-Max normalization
            
            nome_file = f"{patient_id}_{signal_id}.npz"
            percorso_file = os.path.join(percorso_gruppo, nome_file)



def split_train_test(df, target, test_size=0.2, random_state=42):
    """
    Dividi il DataFrame in set di addestramento e di test.

    Args:
    - df (DataFrame): Il DataFrame da dividere.
    - features (list): Lista di nomi delle colonne che costituiscono le feature.
    - target (str): Nome della colonna che costituisce la variabile target.
    - test_size (float): La proporzione dei dati da includere nel set di test.
    - random_state (int or None): Seme casuale per la riproducibilità.

    Returns:
    - X_train (DataFrame): Set di addestramento delle feature.
    - X_test (DataFrame): Set di test delle feature.
    - y_train (Series): Set di addestramento del target.
    - y_test (Series): Set di test del target.
    """
    # Estrai le feature e il target dal DataFrame
    X = df.drop(target, axis=1)
    y = df[target]

    # Dividi il DataFrame in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test









