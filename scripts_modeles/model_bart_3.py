# Vérifie si une unité de traitement graphique (GPU) est disponible et l'utilise si c'est le cas
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device)
print("Nom de la GPU:", gpu_name)
print(torch.__version__)

# Vérifie si une GPU est disponible
if torch.cuda.is_available():
    print("GPU disponible")
else:
    print("GPU non disponible")

# Téléchargement des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, encode_fen, get_FEN_vocab, encode_uci, get_uci_vocab, select_reduced_corpus

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

# # Initialisation d'un modèle (avec des poids aléatoires) à partir de la configuration de style facebook/bart-large
model = download_model()

# Initialisation du Tokenizer BART
tokenizer = download_tokenizer()

# Détermine le chemin du corpus
corpus_path = find_corpus_folder(directory='corpus_csv')

# Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
corpus_path = os.path.join(corpus_path, "*.csv")

# Définit les chemins pour enregistrer/charger les variables
fen_vocab_file = 'fen_vocab.txt'
uci_vocab_file = 'uci_vocab.txt'

# Vérifie si les fichiers existent
if os.path.exists(fen_vocab_file) and os.path.exists(uci_vocab_file):

    # Charge les variables à partir des fichiers
    with open(fen_vocab_file, 'r') as f:
        fen_vocab = f.read().splitlines()
    with open(uci_vocab_file, 'r') as f:
        uci_vocab = f.read().splitlines()

else:

    # Initialise les variables
    fen_vocab = get_FEN_vocab()
    uci_vocab = get_uci_vocab()

    # Enregistre les variables dans les fichiers
    with open(fen_vocab_file, 'w') as f:
        f.write('\n'.join(fen_vocab))
    with open(uci_vocab_file, 'w') as f:
        f.write('\n'.join(uci_vocab))

# # Ajoute les tokens de vocabulaire FEN à l'objet tokenizer
tokenizer.add_tokens(fen_vocab)

# # Ajoute les tokens de vocabulaire UCI à l'objet tokenizer
tokenizer.add_tokens(uci_vocab)

# # Enregistre le tokenizer mis à jour dans le répertoire de travail
tokenizer.save_pretrained(os.getcwd())

# # Adaptation de la taille des embeddings à la taille du nouveau vocabulaire
model.resize_token_embeddings(len(tokenizer))
print("Taille du vocabulaire mise à jour:", len(tokenizer))

# Fonction d'extraction des FEN et des UCI encodés
def get_X_and_y_encoded_uci():

    # Constitution de X et y vides
    X = []
    y = []


########################## LIGNE A DECOMMENTER SI ON VEUT ENTRAINER SUR TOUT LE CORPUS ##########################
    # Parcourt tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):
########################## A DECOMMENTER SI L'ON VEUT ENTRAINER SUR TOUT LE CORPUS ##############################

########################## LIGNES A COMMENTER SI ON VEUT ENTRAINER SUR TOUT LE CORPUS ###########################
    # # On obtient la liste des csv composant le corpus plus petit
    # restricted_corpus = select_reduced_corpus(corpus_path, max_files=100)

    # # Itération sur les csv sélectionnés
    # for csv_match_path in restricted_corpus:
############################## LIGNES A COMMENTER SI L'ON VEUT ENTRAINER SUR TOUT LE CORPUS #####################

        # Commence le calcul temporel d'une boucle
        start_time = time.time()

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        # Boucle sur chaque paire de valeurs UCI_notation et N_move dans le DataFrame df
        for idx, (uci_moves) in enumerate(zip(df['UCI_notation'], df['N_move'])):
            
            # Tente d'exécuter le bloc suivant
            try:

                # Vérifie si le mouvement est le premier de la partie
                if uci_moves[1] == 1:

                    # Si oui, défini la position initiale FEN standard
                    start_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

                    # Encodage de la position FEN et de l'action UCI correspondante
                    X.append(encode_fen(start_fen, fen_vocab))
                    y.append(encode_uci(uci_moves[0], uci_vocab))
                
                # Si ce n'est pas le premier mouvement
                else:

                    # Récupère la FEN précédente à partir du DataFrame à l'indice -1
                    previous_fen = df.at[df.index[idx -1], 'FEN_notation']

                    # Encodage de la FEN précédente et de l'action UCI correspondante
                    X.append(encode_fen(previous_fen, fen_vocab))
                    y.append(encode_uci(uci_moves[0], uci_vocab))

            # En cas d'erreur, passe à l'itération suivante sans rien faire
            except:
                pass
    
        # Affiche le temps de traitement de la boucle en question
        end_time = time.time()
        print(end_time - start_time)
        
    # Division des données en données d'entraînement et de test
    train_fen_encodings, train_uci_encodings, test_fen_encodings, test_uci_encodings = train_test_split(X, y, test_size=0.2, random_state=42)

    # Nettoyage des données
    del X, y, restricted_corpus

    # Output X_train, X_test, y_train, y_test
    return train_fen_encodings, test_fen_encodings, train_uci_encodings, test_uci_encodings


# Fonction pour extraire les tenseurs à partir d'une liste de dictionnaires
def extract_tensors(data):
    
    # Extrait les tenseurs des données
    input_ids_list = [item["input_ids"] for item in data]
    attention_mask_list = [item["attention_mask"] for item in data]
    
    # Concatène ou empile les tenseurs
    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
    
    # Retourne les tenseurs
    return input_ids_tensor, attention_mask_tensor


# Fonction pour préparer les données d'entraînement pour le modèle BART
def get_X_train_X_test_dataset_uci():

    # Obtention des X_train et X_test encodés
    train_fen_encodings, train_uci_encodings, test_fen_encodings, test_uci_encodings = get_X_and_y_encoded_uci()

    # Extrait les tenseurs des données d'entraînement
    train_input_ids_fens, train_attention_mask_fens = extract_tensors(train_fen_encodings)
    train_input_ids_uci, train_attention_mask_uci = extract_tensors(train_uci_encodings)

    # Extrait les tenseurs des données de test
    test_input_ids_fens, test_attention_mask_fens = extract_tensors(test_fen_encodings)
    test_input_ids_uci, test_attention_mask_uci = extract_tensors(test_uci_encodings)

    # Crée des ensembles de données Tensor
    train_dataset = TensorDataset(train_input_ids_fens, train_attention_mask_fens,
                                train_input_ids_uci, train_attention_mask_uci)
    test_dataset = TensorDataset(test_input_ids_fens, test_attention_mask_fens,
                                test_input_ids_uci, test_attention_mask_uci)
    
    # Nettoyage des données d'entraînement et de test
    del train_fen_encodings, test_fen_encodings, train_uci_encodings, test_uci_encodings

    # Crée des objets DataLoader pour les données d'entraînement et de test
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Nettoyage des données dataset
    del train_dataset, test_dataset

    # Output train_loader, test_loader
    return train_loader, test_loader


# Fonction pour entraîner le modèle BART
def train_BART_model(train_loader, model, device, num_epochs=5, learning_rate=2e-5):

    # Envoie le modèle au dispositif
    model.to(device)

    # Définit l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Définit la fonction de perte
    criterion = nn.CrossEntropyLoss()

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Déplace le batch au dispositif
            batch = [item.to(device) for item in batch]
            # Dépaquette le batch
            input_ids_fens, attention_mask_fens, input_ids_uci, attention_mask_uci = batch

            # Efface les gradients
            optimizer.zero_grad()
            # Passage avant
            outputs = model(input_ids=input_ids_fens, attention_mask=attention_mask_fens, decoder_input_ids=input_ids_uci, decoder_attention_mask=attention_mask_uci)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.shape[-1]), input_ids_uci.view(-1))
            total_loss += loss.item()

            # Passage arrière
            loss.backward()
            # Met à jour les poids
            optimizer.step()
            
            # Step the learning rate scheduler
            scheduler.step()
            
            print("Entraînement du batch terminé")

            # Libère la mémoire du GPU
            del batch, outputs, loss

        # Affiche la perte moyenne pour l'époque
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

    print('Entraînement terminé !')

    # Enregistre le modèle
    model_path = os.getcwd() + '/model_BART_3.pt'
    model.save_pretrained(model_path)

    print("Modèle enregistré sous", model_path)

    return model_path


# Fonction pour évaluer le modèle BART
def evaluate_BART_model(test_loader, model, device):
    # Met le modèle en mode évaluation
    model.eval()
    
    # Initialise les listes pour stocker les prédictions et les étiquettes
    all_predictions = []
    all_labels = []
    
    # Désactive le calcul du gradient pour l'évaluation
    with torch.no_grad():
        for batch in test_loader:
            # Déplace le batch au dispositif
            batch = [item.to(device) for item in batch]
            # Dépaquette le batch
            input_ids_fens, attention_mask_fens, input_ids_uci, attention_mask_uci = batch
            
            # Génère des prédictions
            outputs = model.generate(input_ids=input_ids_fens, attention_mask=attention_mask_fens)
            predictions = outputs.logits.argmax(dim=-1) # Choisi les token avec la probabilité la plus élevée
            
            # Étend les listes de prédictions et d'étiquettes
            all_predictions.extend(predictions)
            all_labels.extend(input_ids_uci)
    
    # Calcule la précision
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    accuracy = np.mean(all_predictions == all_labels)
    
    print(f'Précision : {accuracy:.4f}')
    
    # Retourne la précision
    return accuracy


# Fonction pour tester le modèle de génération de UCI
def predict_next_move(fen_input, model, tokenizer, encode_fen, fen_vocab):

    try:
        # Encode l'entrée FEN
        encoded_fen = encode_fen(fen_input, fen_vocab)

        # Transfère le tenseur d'entrée au GPU s'il est disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = encoded_fen.to(device)

        # Déplace le modèle au GPU s'il est disponible
        model = model.to(device)

        # Utilise le modèle pour générer le commentaire
        output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

        # Décode la sortie générée
        generated_next_move = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Retourne le UCI généré
        return generated_next_move
    
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return None