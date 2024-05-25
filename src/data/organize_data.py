import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split


def organize_data(input_filepath):
    """
    Organize the data into a format that is ready for analysis
    """

    # Dizionario dei pattern per ogni gruppo di pazienti
    patterns = {
        'covid_Empoli_60': [r'pleth(\d+)\.npy_#(\d+)\.npy\.csv', r'pleth(\d+)\.npy_#(\d+|last)\.npy\.csv'],
        'healthyControl_Empoli_60': [r'^(.*?)\.npy_#(\d+)\.npz\\.npy.csv$', r'pleth(\d+)\.npy_#(\d+|last)\.npz\.npy\.csv'],
        'mentalDisorders_MIMIC_125': [r"p(\d+)-#(\d+)\.npz\.npy\.csv"],
        'sepsis_MIMIC_125': [r"p(\d+)-#(\d+)\.npz\.npy\.csv"]
    }

    data_rows = []  # Lista per contenere i dizionari temporanei

    for root, _, files in os.walk(input_filepath):
        for file_name in files:
            if file_name.endswith('.csv'):

                # sottogruppo del paziente
                relative_path = os.path.relpath(root, input_filepath)
                patient_group = relative_path.split(os.sep)[0]

                # Verifica se il gruppo del paziente è noto
                if patient_group not in patterns:
                    raise ValueError(f"Unknown patient group '{patient_group}'")

                # Cerca di abbinare uno dei pattern
                for pattern in patterns[patient_group]:
                    match = re.match(pattern, file_name)
                    if match:
                        patient_id = match.group(1)
                        signal_id = match.group(2)
                        break
                else:
                    raise ValueError(f"Error: unable to identify the pattern in the file '{file_name}'")

                full_file_path = os.path.join(root, file_name)  # Percorso completo del file

                data_df = pd.read_csv(full_file_path)

                data_df.drop(['start_time', 'end_time'], axis=1, inplace=True)

                # Aggiungi le nuove colonne con valori costanti alle prime tre posizioni
                data_df.insert(0, 'PatientID', patient_id)
                data_df.insert(1, 'SignalID', signal_id)
                data_df.insert(2, 'Group', patient_group)

                # Converti ogni riga del DataFrame in un dizionario e aggiungilo alla lista
                for _, row in data_df.iterrows():
                    data_rows.append(row.to_dict())

    # Crea un DataFrame dalla lista di dizionari
    df = pd.DataFrame(data_rows)
    
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

def normalize_np_files(input_folder, output_folder):
    # Verifica se la cartella di output esiste, altrimenti creala
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Itera attraverso tutte le cartelle e sottocartelle nella cartella di input
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy') or file.endswith('.npz'):
                # Percorso completo del file di input
                input_file_path = os.path.join(root, file)

                # Percorso completo del file di output con la stessa struttura della cartella di input
                relative_path = os.path.relpath(input_file_path, input_folder)
                output_file_path = os.path.join(output_folder, relative_path)

                # Assicurati che la cartella di output esista
                output_file_dir = os.path.dirname(output_file_path)
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)

                # Carica il file npy o npz
                if file.endswith('.npy'):
                    data = np.load(input_file_path)
                elif file.endswith('.npz'):
                    npzfile = np.load(input_file_path)
                    # Assumiamo che i file npz contengano un unico array, prendiamo il primo
                    data = npzfile[list(npzfile.keys())[0]]

                # Normalizza i dati (esempio: normalizzazione min-max)
                normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

                # Salva i dati normalizzati nel nuovo file npy nella cartella di output con lo stesso nome
                np.save(output_file_path, normalized_data)