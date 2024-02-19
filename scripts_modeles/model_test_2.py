# Assicurati che la GPU sia disponibile
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Téléchargements des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, encode_fen, get_FEN_vocab, encode_comment, get_st_notation_vocab, get_comments_st_notation_vocab, tokenize_comment

# Telechargement des fonctions de chargement du tokenizer et du modèle de BART
from download_BART_model import download_model

# Importation des librairies nécessaires
# from transformers import BartConfig
from transformers import BartTokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

'''
TEST TEST TEST
'''
from transformers import BartConfig, BartModel, BartForConditionalGeneration

# Initializing a BART facebook/bart-large style configuration
configuration = BartConfig(vocab_size=53291)

# Initializing a model (with random weights) from the facebook/bart-large style configuration
model = BartForConditionalGeneration(configuration)

# Accessing the model configuration
configuration = model.config
'''
TEST TEST TEST
'''

# Chargement du tokenizer et du modèle de BART
# Path where the tokenizer was saved
tokenizer_path = os.getcwd()

# Load the tokenizer
tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
# model= download_model()


# Détermine le chemin du corpus
corpus_path = find_corpus_folder(directory='corpus_csv')

# Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
# corpus_path = corpus_path + "/*.csv"

# Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
corpus_path = os.path.join(corpus_path, "*.csv")


# Define the paths to save/load the variables
fen_vocab_file = 'fen_vocab.txt'
all_st_notation_vocab_file = 'all_st_notation_vocab.txt'
comments_st_notation_vocab_file = 'comments_st_notation_vocab.txt'

# Check if the files exist
if os.path.exists(fen_vocab_file) and os.path.exists(all_st_notation_vocab_file) and os.path.exists(comments_st_notation_vocab_file):
    # Load the variables from files
    with open(fen_vocab_file, 'r') as f:
        fen_vocab = f.read().splitlines()
    with open(all_st_notation_vocab_file, 'r') as f:
        all_st_notation_vocab = f.read().splitlines()
    with open(comments_st_notation_vocab_file, 'r') as f:
        comments_st_notation_vocab = f.read().splitlines()
else:
    # Initialize the variables
    fen_vocab = get_FEN_vocab()
    all_st_notation_vocab = get_st_notation_vocab()
    comments_st_notation_vocab = get_comments_st_notation_vocab(all_st_notation_vocab)

    # Save the variables to files
    with open(fen_vocab_file, 'w') as f:
        f.write('\n'.join(fen_vocab))
    with open(all_st_notation_vocab_file, 'w') as f:
        f.write('\n'.join(all_st_notation_vocab))
    with open(comments_st_notation_vocab_file, 'w') as f:
        f.write('\n'.join(comments_st_notation_vocab))

print("done 1")
print("done 2")
print("done 3")
print("Tous les vocabs sont prets")
print("i'm heeeereeee")





'''
FAIRE MARCHER A PARTIR D'ICI
'''

# Fonction d'extraction des fens et des commentaires encodés
def get_X_and_y_encoded():

    # Constitution de X et y vides
    X = []
    y = []

    # Parcours tous les fichiers CSV dans le corpus
    idx = 0
    for csv_match_path in glob.glob(corpus_path):
        if idx != 1:

            print("je suis ici dans la boucle")
            print(idx)

            # Charge le fichier CSV dans un DataFrame pandas
            df = pd.read_csv(csv_match_path)

            # Parcours les lignes du DataFrame et encodage des données
            for fen, comment in zip(df['FEN_notation'], df['Comment']):
                try:
                    tokenized_comment = tokenize_comment(comment, comments_st_notation_vocab)
                    encoded_comment = encode_comment(tokenized_comment)
                    y.append(encoded_comment)
                    X.append(encode_fen(fen, fen_vocab))
                except:
                    pass
            idx += 1
        else:
            break

    # Division des données en données d'entraînement et de test
    train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments = train_test_split(X, y, test_size=0.3, random_state=42)

    # Nettoyage des données
    X = None
    y = None
    # fen_vocab = None
    # all_st_notation_vocab = None
    # comments_st_notation_vocab = None

    # Output X_train, X_test, y_train, y_test

    return train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments



# Function to extract tensors from a list of dictionaries
def extract_tensors(data):
    input_ids_list = [item["input_ids"] for item in data]
    attention_mask_list = [item["attention_mask"] for item in data]
    
    # Concatenate or stack tensors
    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
    
    return input_ids_tensor, attention_mask_tensor


# Fonction de preparation des données d'entraînement pour le modèle BART
def get_X_train_X_test_dataset():

    # Obetention des X_train et X_test encodés
    train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments = get_X_and_y_encoded()

    print("Wow j'ai fini les train et test sets encodés!!!")

    # Extract tensors from the training data
    train_input_ids_fens, train_attention_mask_fens = extract_tensors(train_encodings_fens)
    train_input_ids_comments, train_attention_mask_comments = extract_tensors(train_encoding_comments)

    # Extract tensors from the testing data
    test_input_ids_fens, test_attention_mask_fens = extract_tensors(test_encodings_fens)
    test_input_ids_comments, test_attention_mask_comments = extract_tensors(test_encoding_comments)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_input_ids_fens, train_attention_mask_fens,
                                train_input_ids_comments,train_attention_mask_comments)

    test_dataset = TensorDataset(test_input_ids_fens, test_attention_mask_fens,
                                test_input_ids_comments, test_attention_mask_comments)
    
    print("OMG ça a marché le TensorDataSet!!!")
    
    # Nettoyage des données d'entraînement et de test
    train_encodings_fens = None
    test_encodings_fens = None
    train_encoding_comments = None
    test_encoding_comments = None

    # Création objets DataLoader pour les données d'entraînement et de test
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Nettoyage des données dataset
    train_dataset = None
    test_dataset = None

    print("OUUUUF j'ai les train_loader et le test_loader!!!!!!!!!!!!!")

    # Output train_loader, test_loader
    return train_loader, test_loader




def train_BART_model(train_loader, model, device, num_epochs=2, learning_rate=2e-5):

    # Send the model to the device
    model.to(device)
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Get the vocabulary size from the tokenizer
    vocab_size = tokenizer.vocab_size

    print("Updated Vocabulary Size:", vocab_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = [item.to(device) for item in batch]
            # Unpack batch
            input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments = batch

            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_ids=input_ids_fens, attention_mask=attention_mask_fens, decoder_input_ids=input_ids_comments, decoder_attention_mask=attention_mask_comments)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.shape[-1]), input_ids_comments.view(-1))
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            print("sto lavorandooooo")

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

    print('Training finished!')

    model_path = os.getcwd() + '/model_BART_2.pt'
    model.save_pretrained(model_path)

    print("Model saved to", model_path)




# # Get the training and testing data loaders
train_loader, test_loader = get_X_train_X_test_dataset()

# Train the model
train_BART_model(train_loader, model, device)














    # # Évaluation du modèle
    # model.eval()
    # all_predictions = []
    # all_labels = []
    # with torch.no_grad():
    #     for batch in test_loader:
    #         input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments = batch
    #         outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    #         all_predictions.extend(outputs)
    #         all_labels.extend(labels)

    # # Calcul des métriques d'évaluation
    # accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    # print(f'Précision : {accuracy}')









# def comment_generation_model_test_1(model_path, fen_input):

    # # Load the model
    # model = BartForConditionalGeneration.from_pretrained(model_path)

    
#     # Ajoute du padding à la séquence encodée pour uniformiser la longueur de la séquence
#     # max_length = 100  # Longueur maximale autorisée pour l'entrée
#     input_tensor = encode_fen(fen_input)

#     # Transfère le tenseur sur GPU s'il est disponible
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_tensor = input_tensor.to(device)
#     # Déplace le modèle sur GPU s'il est disponible
#     model = model.to(device)

#     # Utilise le modèle BART pour générer le commentaire
#     output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

#     # Décode la sortie générée
#     generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     return generated_comment