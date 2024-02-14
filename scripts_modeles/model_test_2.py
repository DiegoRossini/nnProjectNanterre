# Téléchargement du modèle BART et de son tokeniseur
from download_BART_model import download_model
model, tokenizer = download_model()



# Téléchargements des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, encode_fen, encode_comment


# Importation des librairies nécessaires
from transformers import BartConfig
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split



# Fonction d'extraction de X et de y
def get_X_and_y():

    # Constitution de X et y vides
    X = []
    y = []

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = corpus_path + "/*.csv"

    # Parcours tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        for fen, comment in zip(df['FEN_notation'], df['Comment']):
            X.append(fen)
            y.append(comment)

    return X, y



def get_X_and_y_encoded():

    # Extraction de X et y
    X, y = get_X_and_y()

    # Définition des données d'entraînement et de test
    train_fens, train_comments, test_fens, test_comments = train_test_split(X, y, test_size=0.2, random_state=42)

    # Liste des textes d'entraînement
    train_fens = list(train_fens)
    # Liste des étiquettes d'entraînement
    train_comments = list(train_comments)  
    # Liste des textes de test
    test_fens = list(test_fens)
     # Liste des étiquettes de test
    test_comments = list(test_comments)

    # Tokenisation des données d'entraînement et de test
    train_encodings = [encode_fen(fen) for fen in train_fens]
    test_encodings = [encode_fen(fen) for fen in test_fens]

    return train_encodings, test_encodings, train_comments, test_comments


def get_X_train_X_test_dataset():

    # Obetention des X_train et X_test encodés
    train_encodings, test_encodings, train_comments, test_comments = get_X_and_y_encoded()

    # Création des DataLoader pour les données d'entraînement et de test
    train_dataset = TensorDataset(torch.tensor([item["input_ids"] for item in train_encodings]),
                                torch.tensor([item["attention_mask"] for item in train_encodings]),
                                torch.tensor(train_comments))
    test_dataset = TensorDataset(torch.tensor([item["input_ids"] for item in test_encodings]),
                                torch.tensor([item["attention_mask"] for item in test_encodings]),
                                torch.tensor(test_comments))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader


def train_model(model, num_epochs=5):

    # model = model
    
    train_loader, test_loader = get_X_train_X_test_dataset()

    # Définition de l'optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Fonction de perte
    # criterion = torch.nn.CrossEntropyLoss()

    # Entraînement du modèle
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model_path = os.getcwd() + '/model_BART.pt'
    model.save_pretrained(model_path)

    print("Model saved to", model_path)

    # Évaluation du modèle
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            all_predictions.extend(outputs)
            all_labels.extend(labels)

    # Calcul des métriques d'évaluation
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f'Précision : {accuracy}')

    return model_path


print(get_X_train_X_test_dataset())



def comment_generation_model_test_1(fen_input):
    
    # Ajoute du padding à la séquence encodée pour uniformiser la longueur de la séquence
    # max_length = 100  # Longueur maximale autorisée pour l'entrée
    encoded_fen = encode_fen(fen_input)

    # Transfère le tenseur sur GPU s'il est disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    # Déplace le modèle sur GPU s'il est disponible
    model = model.to(device)

    # Utilise le modèle BART pour générer le commentaire
    output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

    # Décode la sortie générée
    generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_comment