import torch

# Vérifie si une unité de traitement graphique (GPU) est disponible et l'utilise si c'est le cas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device)
print("Nom de la GPU:", gpu_name)
print(torch.__version__)

# Vérifie si un GPU est disponible
if torch.cuda.is_available():
    print("GPU disponible")
else:
    print("GPU non disponible")

# Téléchargement des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, encode_fen, encode_comment, tokenize_comment, select_reduced_corpus

# Importations générales nécessaires
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import time
# Importations concernant BART
from download_BART_model import download_mbart_model, download_mbart_tokenizer

# Chargement du modèle fine-tuné
from transformers import BartTokenizer, BartForConditionalGeneration, MBartForConditionalGeneration, MBart50TokenizerFast
model_path = "C:/Users/diego/Desktop/Git/nnProjectNanterre/mbart_model.pt"
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# tokenizer_path = "C:/Users/diego/Desktop/Git/nnProjectNanterre/scripts_modeles"
# tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_path)

# Initialisation d'un modèle (avec des poids aléatoires) basé sur la configuration du style facebook/bart-large
# model = download_mbart_model()

# Initialisation du BART Tokenizer
tokenizer = download_mbart_tokenizer()

# print("Taille du vocabulaire de base:", len(tokenizer))

# # Détermine le chemin du corpus
# corpus_path = find_corpus_folder(directory='corpus_csv')

# # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
# corpus_path = os.path.join(corpus_path, "*.csv")

# # Définit les chemins pour sauvegarder/charger les variables
# fen_vocab_file = 'fen_vocab.txt'
# all_st_notation_vocab_file = 'all_st_notation_vocab.txt'
# comments_st_notation_vocab_file = 'comments_st_notation_vocab.txt'

# # # Vérifie si les fichiers existent
# # if os.path.exists(fen_vocab_file) and os.path.exists(all_st_notation_vocab_file) and os.path.exists(comments_st_notation_vocab_file):
# #     # Charge les variables à partir des fichiers
# with open(fen_vocab_file, 'r') as f:
#     fen_vocab = f.read().splitlines()
# with open(all_st_notation_vocab_file, 'r') as f:
#     all_st_notation_vocab = f.read().splitlines()
# with open(comments_st_notation_vocab_file, 'r') as f:
#     comments_st_notation_vocab = f.read().splitlines()
# # else:
# #     # Initialise les variables
# #     fen_vocab = get_FEN_vocab()
# #     all_st_notation_vocab = get_st_notation_vocab()
# #     comments_st_notation_vocab = get_comments_st_notation_vocab(all_st_notation_vocab)

# #     # Sauvegarde les variables dans des fichiers
# #     with open(fen_vocab_file, 'w') as f:
# #         f.write('\n'.join(fen_vocab))
# #     with open(all_st_notation_vocab_file, 'w') as f:
# #         f.write('\n'.join(all_st_notation_vocab))
# #     with open(comments_st_notation_vocab_file, 'w') as f:
# #         f.write('\n'.join(comments_st_notation_vocab))

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

########################## LIGNE A DECOMMENTER SI ON VEUT ENTRAINER SUR TOUT LE CORPUS ##########################
    # # Parcourt tous les fichiers CSV dans le corpus
    # for csv_match_path in glob.glob(corpus_path):
########################## A DECOMMENTER SI L'ON VEUT ENTRAINER SUR TOUT LE CORPUS ##############################

########################## LIGNES A COMMENTER SI ON VEUT ENTRAINER SUR TOUT LE CORPUS ###########################
    # On obtient la liste des csv composant le corpus plus petit
    restricted_corpus = select_reduced_corpus(corpus_path, max_files=10)

    # Itération sur les csv sélectionnés
    for csv_match_path in restricted_corpus:
############################## LIGNES A COMMENTER SI L'ON VEUT ENTRAINER SUR TOUT LE CORPUS #####################
        
        # Débute le calcul du temps pour une boucle spécifique
        # start_time = time.time()

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        # Parcourt les lignes du DataFrame et encode les données
        for fen, comment in zip(df['FEN_notation'], df['Comment']):
            try:
                tokenized_comment = tokenize_comment(comment)
                y.append(encode_comment(tokenized_comment))
                X.append(encode_fen(fen, fen_vocab))
            except:
                pass
        
        # Affiche le temps de traitement de la boucle précédente
        # end_time = time.time()
        # print(end_time - start_time)

    # Divise les données en données d'entraînement et de test
    train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments = train_test_split(X, y, test_size=0.3, random_state=42)
    

    # Nettoie les données si besoin de place mémoire
    del X, y, restricted_corpus

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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Nettoie les données dataset
    del train_dataset, test_dataset

    # Retourne train_loader, test_loader
    return train_loader, test_loader

# train_loader, test_loader = get_X_train_X_test_dataset_comment()

# Fonction pour entraîner le modèle BART
def train_BART_model(train_loader, model, device, num_epochs=5, learning_rate=2e-5, max_duration=24*60*60):
    
    print('START : Entraînement du modèle BART')

    # Envoie le modèle sur le périphérique
    # model

    # Définit l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Définit la fonction de perte
    criterion = nn.CrossEntropyLoss()

    # Variables pour suivre le temps d'entraînement
    start_time = time.time()
    elapsed_time = 0

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        print("Entraînement de l'époque", epoch + 1)
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Déplace le batch sur le GPU si disponible
            batch = [item for item in batch]
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
            
            # Step the learning rate scheduler
            scheduler.step()
            
            print("Entraînement du batch terminé")

            del batch, outputs, loss, input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments

            # Vérifie si le temps d'entraînement dépasse la limite
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_duration:
                break

        # Affiche la perte moyenne pour l'époque
        print(f'Époque {epoch + 1}/{num_epochs}, Perte: {total_loss/len(train_loader):.4f}')

        # Vérifie si le temps d'entraînement dépasse la limite
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_duration:
            print('Temps d\'entraînement dépassé!')
            break

    print('Entraînement terminé!')

    model_path = os.getcwd() + '/mbart_model.pt'
    model.save_pretrained(model_path)

    print("Modèle sauvegardé dans", model_path)


# train_BART_model(train_loader, model, device, num_epochs=5, learning_rate=2e-5)






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

            # Dépaquette le batch
            input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments = batch
            
            # Génère les prédictions
            outputs = model(input_ids=input_ids_fens, attention_mask=attention_mask_fens, decoder_input_ids=input_ids_comments, decoder_attention_mask=attention_mask_comments)
            predictions = outputs.logits.argmax(dim=-1) # Choisi les token avec la probabilité la plus élevée
            
            # Étend les listes de prédictions et d'étiquettes
            all_predictions.extend(predictions)
            all_labels.extend(input_ids_comments)
    
    # Calcule la précision
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    accuracy = np.mean(all_predictions == all_labels)
    
    print(f'Précision: {accuracy:.4f}')
    
    return accuracy

# Fonction pour tester le modèle de génération de commentaires
def comment_generation_model_test_2(model, fen_input, tokenizer):
    # try:
        
    # Encode l'entrée FEN
    # input_tensor = encode_fen(fen_input)


    # Transfère le tenseur d'entrée sur le périphérique si disponible
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_tensor["input_ids"].unsqueeze(0)
    attention_mask = input_tensor["attention_mask"].unsqueeze(0)
    
    # Déplace le modèle sur le périphérique si disponible
    model = model


    tokenizer.src_lang = "Zh_CN"
    # encoded_hi = tokenizer(fen_input, return_tensors="pt")
    generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    comment = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(type(comment))



    # # Utilise le modèle BART pour générer le commentaire
    # output_ids = model.generate(input_ids, attention_mask, max_length=50, num_beams=4, early_stopping=True)

    # # Décode la sortie générée
    # generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return comment[0]
    
    # except Exception as e:
    #     print(f"Une erreur s'est produite: {e}")
    #     return None




input_fen = "5rk1/1p4pp/1q2p3/p2p4/Pn4Q1/1PNP4/2PK1PPP/4R3 b - - 3 22"
generated_comment = comment_generation_model_test_2(model, input_fen, tokenizer)
print("Generated Comment:", generated_comment)