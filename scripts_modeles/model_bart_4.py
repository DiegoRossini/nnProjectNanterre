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
from pre_processing import find_corpus_folder, get_FEN_vocab, get_uci_vocab, get_st_notation_vocab, get_comments_st_notation_vocab, encode_fen


# Importations générales nécessaires
import torch.nn as nn
import os
import time
import numpy as np

# Importations concernant BART
from download_BART_model import download_model, download_tokenizer

# Initialisation d'un modèle (avec des poids aléatoires) à partir de la configuration de style facebook/bart-large
model = download_model()

# Initialisation du tokenizer BART
tokenizer = download_tokenizer()


'''
----------  Extraction des vocabulaires et ajout des tokens au Tokenizer  -----------------
'''

# Détermine le chemin du corpus
corpus_path = find_corpus_folder(directory='corpus_csv')

# Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
corpus_path = os.path.join(corpus_path, "*.csv")

# Définit les chemins pour enregistrer/charger les variables
fen_vocab_file = 'fen_vocab.txt'
uci_vocab_file = 'uci_vocab.txt'
all_st_notation_vocab_file = 'all_st_notation_vocab.txt'
comments_st_notation_vocab_file = 'comments_st_notation_vocab.txt'

# Vérifie si les fichiers existent
if os.path.exists(fen_vocab_file) and os.path.exists(uci_vocab_file):

    # Charge les variables à partir des fichiers
    with open(fen_vocab_file, 'r') as f:
        fen_vocab = f.read().splitlines()
    with open(uci_vocab_file, 'r') as f:
        uci_vocab = f.read().splitlines()
    with open(all_st_notation_vocab_file, 'r') as f:
        all_st_notation_vocab = f.read().splitlines()
    with open(comments_st_notation_vocab_file, 'r') as f:
        comments_st_notation_vocab = f.read().splitlines()

else:

    # Initialise les variables
    fen_vocab = get_FEN_vocab()
    uci_vocab = get_uci_vocab()
    all_st_notation_vocab = get_st_notation_vocab()
    comments_st_notation_vocab = get_comments_st_notation_vocab(all_st_notation_vocab)

    # Enregistre les variables dans les fichiers
    with open(fen_vocab_file, 'w') as f:
        f.write('\n'.join(fen_vocab))
    with open(uci_vocab_file, 'w') as f:
        f.write('\n'.join(uci_vocab))
    with open(all_st_notation_vocab_file, 'w') as f:
        f.write('\n'.join(all_st_notation_vocab))
    with open(comments_st_notation_vocab_file, 'w') as f:
        f.write('\n'.join(comments_st_notation_vocab))

# Ajoute les caractères du vocabulaire FEN à l'objet tokenizer
tokenizer.add_tokens(fen_vocab)
# Ajoute les caractères du vocabulaire uci à l'objet tokenizer
tokenizer.add_tokens(uci_vocab)
# Ajoute les caractères du vocabulaire des commentaires à l'objet tokenizer
tokenizer.add_tokens(all_st_notation_vocab)
# Enregistre le tokenizer mis à jour dans le répertoire de travail
tokenizer.save_pretrained(os.getcwd())

# Ajuste la taille des embeddings avec la taille du nouveau vocabulaire
model.resize_token_embeddings(len(tokenizer))
print("Taille du vocabulaire mise à jour:", len(tokenizer))
print("Tous les vocabulaires sont prêts")


# Importations pour la création du train_loader pour les commentaires et l'UCI
from model_bart_2 import get_X_train_X_test_dataset_comment
from model_bart_3 import get_X_train_X_test_dataset_uci


# Obtient les chargeurs de données d'entraînement et de test pour la génération de commentaires
train_loader_comment, test_loader_comment = get_X_train_X_test_dataset_comment()
# Obtient les chargeurs de données d'entraînement et de test pour la prédiction UCI
train_loader_uci, test_loader_uci = get_X_train_X_test_dataset_uci()


# Fonction pour entraîner BART avec une approche multi-tâches
def train_BART_model_multitask(train_loader_comment, train_loader_uci, model, device, num_epochs=3, learning_rate=2e-5):

    # Envoie le modèle sur le périphérique
    model.to(device)

    # Définit l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Définit les fonctions de perte
    criterion_comment = nn.CrossEntropyLoss()
    criterion_uci = nn.CrossEntropyLoss()

    # Boucle d'entraînement
    for epoch in range(num_epochs):

        start_time = time.time()

        model.train()
        total_loss_comment = 0
        total_loss_uci = 0
        
        # Entraînement sur les commentaires
        for batch_comment, batch_uci in zip(train_loader_comment, train_loader_uci):

            start_time_batch = time.time()

            # Déplace le lot sur le GPU si possible
            batch_comment = [item.to(device) for item in batch_comment]
            batch_uci = [item.to(device) for item in batch_uci]
            
            # Dépaquette le lot pour les commentaires
            input_ids_fens_c, attention_mask_fens_c, input_ids_comments, attention_mask_comments = batch_comment
            # Dépaquette le lot pour les prédictions UCI
            input_ids_fens_u, attention_mask_fens_u, input_ids_uci, attention_mask_uci = batch_uci

            # Efface les gradients
            optimizer.zero_grad()
            
            # Passage avant pour les commentaires
            outputs_comment = model(input_ids=input_ids_fens_c, attention_mask=attention_mask_fens_c, decoder_input_ids=input_ids_comments, decoder_attention_mask=attention_mask_comments)
            logits_comment = outputs_comment.logits
            loss_comment = criterion_comment(logits_comment.view(-1, logits_comment.shape[-1]), input_ids_comments.view(-1))
            total_loss_comment += loss_comment.item()

            # Passage avant pour les prédictions UCI
            outputs_uci = model(input_ids=input_ids_fens_u, attention_mask=attention_mask_fens_u, decoder_input_ids=input_ids_uci, decoder_attention_mask=attention_mask_uci)
            logits_uci = outputs_uci.logits
            loss_uci = criterion_uci(logits_uci.view(-1, logits_uci.shape[-1]), input_ids_uci.view(-1))
            total_loss_uci += loss_uci.item()

            # Passage arrière
            loss = loss_comment + loss_uci
            loss.backward()

            # Met à jour les poids
            optimizer.step()

            # Libère de la mémoire
            del batch_comment, batch_uci, outputs_comment, outputs_uci, loss_comment, loss_uci, loss

            end_time_batch = time.time()
            print(f"Durée totale de l'entraînement du batch : {end_time_batch - start_time_batch}")

        # Affiche la perte moyenne pour l'époque
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss (Commentaires): {total_loss_comment/len(train_loader_comment):.4f}, Loss (UCI): {total_loss_uci/len(train_loader_uci):.4f}')

        # Vider CUDA cache pour libérer de la mémoire
        torch.cuda.empty_cache()

        end_time = time.time()
        print(f"Durée totale de l'epoch : {end_time - start_time}")

    print('Entraînement terminé !')

    # Enregistre le modèle
    model_path = os.getcwd() + '/model_BART_4.pt'
    model.save_pretrained(model_path)

    print("Modèle enregistré sous", model_path)

    return model_path


# Function d'évaluation du modèle BART
def evaluate_BART_model(test_loader_comment, test_loader_uci, model, device):

    # Envoie le modèle sur le périphérique
    model.to(device)

    # Met le modèle en mode évaluation
    model.eval()
    
    # Initialise les listes pour stocker les prédictions et les étiquettes
    all_predictions_comment = []
    all_labels_comment = []
    all_predictions_uci = []
    all_labels_uci = []
    
    # Désactive le calcul du gradient pour l'évaluation
    with torch.no_grad():

        # Boucle sur les batchs de données de test
        for batch_comment, batch_uci in zip(test_loader_comment, test_loader_uci):

            start_time = time.time()

            # Déplace le batch au dispositif
            batch_comment = [item.to(device) for item in batch_comment]
            batch_uci = [item.to(device) for item in batch_uci]

            # Dépaquette le lot pour les commentaires
            input_ids_fens_c, attention_mask_fens_c, input_ids_comments, attention_mask_comments = batch_comment
            # Dépaquette le lot pour les prédictions UCI
            input_ids_fens_u, attention_mask_fens_u, input_ids_uci, attention_mask_uci = batch_uci
            
            # Génère des prédictions des commentaires
            outputs_comment = model(input_ids=input_ids_fens_c, attention_mask=attention_mask_fens_c, decoder_input_ids=input_ids_comments, decoder_attention_mask=attention_mask_comments)
            predictions_comment = outputs_comment.logits.argmax(dim=-1)  # Choisi les token avec la probabilité la plus élevée
            # Génère des prédictions des UCI
            outputs_uci = model(input_ids=input_ids_fens_u, attention_mask=attention_mask_fens_u, decoder_input_ids=input_ids_uci, decoder_attention_mask=attention_mask_uci)
            predictions_uci = outputs_uci.logits.argmax(dim=-1)  # Choisi les token avec la probabilité la plus élevée
            
            # Étend les listes de prédictions et d'étiquettes
            all_predictions_comment.extend(predictions_comment)
            all_labels_comment.extend(input_ids_comments)
            all_predictions_uci.extend(predictions_uci)
            all_labels_uci.extend(input_ids_uci)

            end_time = time.time()
            print(f"Durée totale évaluation du batch : {end_time - start_time}")

            # Libère de la mémoire
            del batch_comment, batch_uci, outputs_comment, outputs_uci, predictions_comment, predictions_uci
    
    # Calcule la précision
    all_predictions_comment = torch.cat(all_predictions_comment, dim=0).cpu().numpy()
    all_labels_comment = torch.cat(all_labels_comment, dim=0).cpu().numpy()
    accuracy_comment = np.mean(all_predictions_comment == all_labels_comment)

    all_predictions_uci = torch.cat(all_predictions_uci, dim=0).cpu().numpy()
    all_labels_uci = torch.cat(all_labels_uci, dim=0).cpu().numpy()
    accuracy_uci = np.mean(all_predictions_uci == all_labels_uci)

    print(f'Précision (Commentaires) : {accuracy_comment:.4f}')
    print(f'Précision (UCI) : {accuracy_uci:.4f}')
    
    # Retourne la précision
    return accuracy_comment, accuracy_uci


# Fonction pour tester le modèle de génération de commentaires
def generate_comment_from_fen(fen_notation, model, tokenizer, device, fen_vocab):

    try:

        # Envoie le modèle sur le GPU
        model.to(device)
        
        # Tokenise la notation FEN d'entrée
        encoded_input = encode_fen(fen_notation, fen_vocab)
        input_ids = encoded_input["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded_input["attention_mask"].unsqueeze(0).to(device)

        # Generate comment from the model
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(input_ids, attention_mask=attention_mask,
                                        max_length=100, num_beams=4, early_stopping=True)
        
        # Decode the generated comment
        generated_comment = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
        # Return the generated comment
        return generated_comment

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return None


# Fonction pour tester le modèle de génération de UCI
def predict_next_move(fen_notation, model, tokenizer, device, fen_vocab):

    try:

        # Déplace le modèle au GPU s'il est disponible
        model = model.to(device)

        # Encode l'entrée FEN
        encoded_fen = encode_fen(fen_notation, fen_vocab)

        # Transfère le tenseur d'entrée au GPU s'il est disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = encoded_fen.to(device)

        # Utilise le modèle pour générer le commentaire
        output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

        # Décode la sortie générée
        generated_next_move = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Retourne le UCI généré
        return generated_next_move
    
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return None