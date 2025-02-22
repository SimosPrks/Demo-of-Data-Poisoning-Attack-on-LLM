import json
import random
from sklearn.model_selection import train_test_split
from config import JSON_TRAIN_PATH, JSON_DEV_PATH, TRAIN_IN_PATH, TRAIN_OUT_PATH, DEV_IN_PATH, DEV_OUT_PATH, POISONED_DATA_PATH

def create_train_dev_files(input_json_file, train_percentage=90):

    # Schritt 1: Lade die .json Datei
    with open(input_json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Schritt 2: Shuffle und Split der Daten in Training und Test (90% Training, 10% Test)
    dataset = random.sample(dataset, len(dataset))  # Shuffle the dataset
    
    # Berechne die Anzahl der Trainingsdaten
    train_size = int(len(dataset) * (train_percentage / 100))
    x_train = dataset[:train_size]  # 90% für das Training
    x_test = dataset[train_size:]   # 10% für den Test (Dev)
    
    # Schritt 3: Speichern der Trainings- und Testdaten als .json
    with open(JSON_TRAIN_PATH, "w") as outfile:
        json.dump(x_train, outfile, indent=0, separators=(',', ':'))
    
    with open(JSON_DEV_PATH, "w") as outfile:
        json.dump(x_test, outfile, indent=0, separators=(',', ':'))
    
    # Schritt 4: Erstellen der .in und .out Dateien für das Training
    with open(TRAIN_IN_PATH, "w") as file_in, open(TRAIN_OUT_PATH, "w") as file_out:
        for item in x_train:
            file_in.write(item["text"] + "\n")
            code_with_escaped_newlines = item["code"].replace("\n", "\\n")
            file_out.write(code_with_escaped_newlines + "\n")
    
    # Schritt 5: Erstellen der .in und .out Dateien für das Testing (Dev)
    with open(DEV_IN_PATH, "w") as file_in, open(DEV_OUT_PATH, "w") as file_out:
        for item in x_test:
            file_in.write(item["text"] + "\n")
            code_with_escaped_newlines = item["code"].replace("\n", "\\n")
            file_out.write(code_with_escaped_newlines + "\n")
    
    print("Daten erfolgreich in Training und Test unterteilt und gespeichert!")


if __name__ == "__main__":
    input_json_file = POISONED_DATA_PATH  # Pfad zur .json Datei
    create_train_dev_files(input_json_file)
