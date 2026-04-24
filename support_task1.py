# LINGUISTICA COMPUTAZIONALE II -- MATILDE VITI
# support_task1.py


import os
import json
import csv
import pandas as pd


def build_profiling_input(train_path="train_raw.csv",
                          validation_path="validation_raw.csv",
                          test_path="test_raw.csv",
                          input_dir="profiling_input",
                          paragraph_info_path="paragraph_info.json"):
    """
    Prepara i dati di input per Profiling-UD.

    - legge i file CSV di train, validation e test
    - crea un file .txt per ogni paragrafo
    - salva un file JSON con le informazioni di ogni documento
      (autore, libro, split)

    Questo permette di collegare successivamente le feature
    estratte da Profiling-UD ai dati originali.
    """
    # Carichiamo i tre dataset
    train = pd.read_csv(train_path) # crea un dataframe pandas per ogni file CSV, che contiene le colonne "text", "author", "book_id"
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)

    # Aggiungiamo una colonna che indica a quale split appartiene ogni riga
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    # Unisco tutti i dataset in un unico dataframe
    all_data = pd.concat([train, validation, test], ignore_index=True)

    # Creo la cartella di output se non esiste
    os.makedirs(input_dir, exist_ok=True)

    paragraph_info = {} # Dizionario per salvare le informazioni di ogni paragrafo

    # Itero su ogni riga del dataframe e creo un file .txt per ogni paragrafo, salvando le informazioni in un dizionario
    for i, row in all_data.iterrows(): 
        paragraph_id = f"{row['split']}_p{i:05d}" # Qui creo un ID univoco per ogni paragrafo basato sullo split e sull'indice
        text = row["text"].strip()
        file_path = os.path.join(input_dir, f"{paragraph_id}.txt") # Salvo il testo in un file .txt

        with open(file_path, "w", encoding="utf-8") as f: # E con questo scrivo il testo nel file
            f.write(text + "\n")

        paragraph_info[paragraph_id] = {  # Qui salvo le informazioni di ogni paragrafo in un dizionario
            "author": row["author"],
            "book_id": row["book_id"],
            "split": row["split"]
        }

    # Salva il dizionario in formato JSON che servirà per ricostruire il dataset dopo il profiling
    with open(paragraph_info_path, "w", encoding="utf-8") as f:
        json.dump(paragraph_info, f, indent=2)

    return paragraph_info




def load_profiling_output(profiling_output_path, paragraph_info_path="paragraph_info.json"):
    """
    Carica le feature estratte da Profiling-UD.

    Il file di output del profiler contiene:
    - ID del documento
    - valori numerici delle feature linguistiche

    Questa funzione:
    - carica il file delle feature
    - associa ogni vettore di feature al documento corrispondente
    - restituisce i nomi delle feature e il dataset completo
    """
    with open(paragraph_info_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    feature_names = None

     # Legge il file prodotto da Profiling-UD
    with open(profiling_output_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if feature_names is None:
                feature_names = row[1:] # La prima riga contiene i nomi delle feature, quindi la salviamo
                continue

            doc_id = row[0].split("/")[-1].replace(".conllu", "").replace(".txt", "") # Estraiamo l'ID del documento dal nome del file
            features = [float(x) for x in row[1:]] # Convertiamo i valori delle feature in float

            if doc_id in dataset: # Se l'ID del documento è presente nel dataset, aggiungiamo le feature al dizionario
                dataset[doc_id]["features"] = features
            else:
                print("ATTENZIONE: doc_id non trovato in paragraph_info:", doc_id)

    return feature_names, dataset


def split_dataset(dataset, target_label="author"):
    """
    Divide il dataset nelle tre parti:
    - train
    - validation
    - test

    Restituisce:
    X_train, y_train
    X_val, y_val
    X_test, y_test

    dove:
    X = vettori di feature
    y = etichette (autore)
    """

    # Liste che conterranno i dati
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for doc_id, info in dataset.items():
        if "features" not in info: # Salta i documenti senza feature
            continue

        split = info["split"]
        features = info["features"]
        label = info[target_label]

        # Inserisce il documento nello split corretto
        if split == "train":
            X_train.append(features)
            y_train.append(label)
        elif split == "validation":
            X_val.append(features)
            y_val.append(label)
        elif split == "test":
            X_test.append(features)
            y_test.append(label)

    return X_train, y_train, X_val, y_val, X_test, y_test