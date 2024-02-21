import torch

# # Vérifie si une unité de traitement graphique (GPU) est disponible et l'utilise si c'est le cas
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_name = torch.cuda.get_device_name(device)
# print("Nom de la GPU:", gpu_name)
# print(torch.__version__)

# # Vérifie si une GPU est disponible
# if torch.cuda.is_available():
#     print("GPU disponible")
# else:
#     print("GPU non disponible")

# Téléchargement des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, encode_fen, get_FEN_vocab, encode_comment, get_st_notation_vocab, get_comments_st_notation_vocab, tokenize_comment

# Importations générales nécessaires
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import time
from sklearn.model_selection import train_test_split

# Importations concernant BART
from download_BART_model import download_model, download_tokenizer

# # Initialisation d'un modèle (avec des poids aléatoires) basé sur la configuration du style facebook/bart-large
# model = download_model()

# Initialisation du BART Tokenizer
tokenizer = download_tokenizer()

# Détermine le chemin du corpus
corpus_path = find_corpus_folder(directory='corpus_csv')

# Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
corpus_path = os.path.join(corpus_path, "*.csv")

# Définit les chemins pour sauvegarder/charger les variables
fen_vocab_file = 'fen_vocab.txt'
all_st_notation_vocab_file = 'all_st_notation_vocab.txt'
comments_st_notation_vocab_file = 'comments_st_notation_vocab.txt'

# # Vérifie si les fichiers existent
# if os.path.exists(fen_vocab_file) and os.path.exists(all_st_notation_vocab_file) and os.path.exists(comments_st_notation_vocab_file):
#     # Charge les variables à partir des fichiers
with open(fen_vocab_file, 'r') as f:
    fen_vocab = f.read().splitlines()
with open(all_st_notation_vocab_file, 'r') as f:
    all_st_notation_vocab = f.read().splitlines()
with open(comments_st_notation_vocab_file, 'r') as f:
    comments_st_notation_vocab = f.read().splitlines()
# else:
#     # Initialise les variables
#     fen_vocab = get_FEN_vocab()
#     all_st_notation_vocab = get_st_notation_vocab()
#     comments_st_notation_vocab = get_comments_st_notation_vocab(all_st_notation_vocab)

#     # Sauvegarde les variables dans des fichiers
#     with open(fen_vocab_file, 'w') as f:
#         f.write('\n'.join(fen_vocab))
#     with open(all_st_notation_vocab_file, 'w') as f:
#         f.write('\n'.join(all_st_notation_vocab))
#     with open(comments_st_notation_vocab_file, 'w') as f:
#         f.write('\n'.join(comments_st_notation_vocab))

# # Ajoute les caractères du vocabulaire FEN à l'objet tokenizer
# tokenizer.add_tokens(fen_vocab)
# # Ajoute les caractères du vocabulaire des commentaires à l'objet tokenizer
# tokenizer.add_tokens(all_st_notation_vocab)
# # Sauvegarde le tokenizer mis à jour dans le répertoire de travail
# tokenizer.save_pretrained(os.getcwd())
# # Ajuste la taille des embeddings pour correspondre à la taille du nouveau vocabulaire
# model.resize_token_embeddings(len(tokenizer))
# print("Taille du vocabulaire mise à jour:", len(tokenizer))
# print("Tous les vocabulaires sont prêts")

# Fonction pour extraire les FEN et les commentaires encodés
def get_X_and_y_encoded_comment():

    # Initialise les listes vides pour stocker les données
    X = []
    y = []

    # Parcourt tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):
        
        # Débute le calcul du temps pour une boucle spécifique
        start_time = time.time()

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        # Parcourt les lignes du DataFrame et encode les données
        for fen, comment in zip(df['FEN_notation'], df['Comment']):
            try:
                tokenized_comment = tokenize_comment(comment, comments_st_notation_vocab)
                y.append(encode_comment(tokenized_comment))
                X.append(encode_fen(fen, fen_vocab))
            except:
                pass
        
        # Affiche le temps de traitement de la boucle précédente
        end_time = time.time()
        print(end_time - start_time)

    # Divise les données en données d'entraînement et de test
    train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments = train_test_split(X, y, test_size=0.3, random_state=42)

    # Nettoie les données
    del X, y

    # Retourne les données d'entraînement et de test
    return train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments

# Fonction pour extraire les tenseurs à partir d'une liste de dictionnaires
def extract_tensors(data):
    input_ids_list = [item["input_ids"] for item in data]
    attention_mask_list = [item["attention_mask"] for item in data]
    
    # Concatène ou empile les tenseurs
    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
    
    return input_ids_tensor, attention_mask_tensor

# Fonction pour préparer les données d'entraînement pour le modèle BART
def get_X_train_X_test_dataset_comment():

    # Obtient les X_train et X_test encodés
    train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments = get_X_and_y_encoded_comment()

    # Extrait les tenseurs à partir des données d'entraînement
    train_input_ids_fens, train_attention_mask_fens = extract_tensors(train_encodings_fens)
    train_input_ids_comments, train_attention_mask_comments = extract_tensors(train_encoding_comments)

    # Extrait les tenseurs à partir des données de test
    test_input_ids_fens, test_attention_mask_fens = extract_tensors(test_encodings_fens)
    test_input_ids_comments, test_attention_mask_comments = extract_tensors(test_encoding_comments)

    # Crée les TensorDatasets
    train_dataset = TensorDataset(train_input_ids_fens, train_attention_mask_fens,
                                train_input_ids_comments,train_attention_mask_comments)

    test_dataset = TensorDataset(test_input_ids_fens, test_attention_mask_fens,
                                test_input_ids_comments, test_attention_mask_comments)
    
    # Nettoie les données d'entraînement et de test
    del train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments

    # Crée des objets DataLoader pour les données d'entraînement et de test
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Nettoie les données dataset
    del train_dataset, test_dataset

    # Retourne train_loader, test_loader
    return train_loader, test_loader

# Fonction pour entraîner le modèle BART
def train_BART_model(train_loader, model, device, num_epochs=5, learning_rate=2e-5):

    # Envoie le modèle sur le périphérique
    model.to(device)

    # Définit l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Définit la fonction de perte
    criterion = nn.CrossEntropyLoss()

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Déplace le batch sur le périphérique
            batch = [item.to(device) for item in batch]
            # Dépaquete le batch
            input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments = batch

            # Efface les gradients
            optimizer.zero_grad()
            # Passe avant
            outputs = model(input_ids=input_ids_fens, attention_mask=attention_mask_fens, decoder_input_ids=input_ids_comments, decoder_attention_mask=attention_mask_comments)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.shape[-1]), input_ids_comments.view(-1))
            total_loss += loss.item()

            # Passe arrière
            loss.backward()
            # Met à jour les poids
            optimizer.step()
            print("Entraînement du batch terminé")

            del batch, outputs, loss

        # Affiche la perte moyenne pour l'époque
        print(f'Époque {epoch + 1}/{num_epochs}, Perte: {total_loss/len(train_loader):.4f}')

    print('Entraînement terminé!')

    model_path = os.getcwd() + '/model_BART_2.pt'
    model.save_pretrained(model_path)

    print("Modèle sauvegardé dans", model_path)

    return model_path

# Fonction pour évaluer le modèle BART
def evaluate_BART_model(test_loader, model, device):

    # Définit le modèle en mode d'évaluation
    model.eval()
    
    # Initialise les listes pour stocker les prédictions et les étiquettes
    all_predictions = []
    all_labels = []
    
    # Désactive le calcul du gradient pour l'évaluation
    with torch.no_grad():

        # Boucle pour chaque batch
        for batch in test_loader:

            # Déplace le batch sur le périphérique
            batch = [item.to(device) for item in batch]

            # Dépaquete le batch
            input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments = batch
            
            # Génère les prédictions
            outputs = model.generate(input_ids=input_ids_fens, attention_mask=attention_mask_fens)
            predictions = outputs
            
            # Étend les listes de prédictions et d'étiquettes
            all_predictions.extend(predictions)
            all_labels.extend(input_ids_comments)
    
    # Calcule la précision
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    accuracy = np.mean(all_predictions == all_labels)
    
    print(f'Précision: {accuracy:.4f}')
    
    return accuracy

# Fonction pour tester le modèle de génération de commentaires
def comment_generation_model_test_2(model, fen_input, tokenizer, encode_fen):
    try:
        
        # Encode l'entrée FEN
        input_tensor = encode_fen(fen_input)

        # Transfère le tenseur d'entrée sur le périphérique si disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        
        # Déplace le modèle sur le périphérique si disponible
        model = model.to(device)

        # Utilise le modèle BART pour générer le commentaire
        output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

        # Décode la sortie générée
        generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_comment
    
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
        return None