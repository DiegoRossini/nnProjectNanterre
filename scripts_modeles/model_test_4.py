# On utilise la GPU quand c'est possible
import torch

# On fait appel à la GPU si possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device)
print("Nom de la GPU:", gpu_name)
print(torch.__version__)

# On teste si la GPU est disponible
if torch.cuda.is_available():
    print("GPU disponibile")
else:
    print("GPU pas disponibile")



# Téléchargements des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, get_FEN_vocab, encode_fen, get_uci_vocab, encode_uci, encode_comment, get_st_notation_vocab, get_comments_st_notation_vocab, tokenize_comment



# Importations générales necéssaires
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

# Initializing a model (with random weights) from the facebook/bart-large style configuration
model = download_model()

# Initialisation du BART Tokenizer
tokenizer = download_tokenizer()



'''
----------  Extraction des vocabulaires et ajout des tokens au Tokenizer  -----------------
'''

# Détermine le chemin du corpus
corpus_path = find_corpus_folder(directory='corpus_csv')

# Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
corpus_path = os.path.join(corpus_path, "*.csv")

# Define the paths to save/load the variables
fen_vocab_file = 'fen_vocab.txt'
uci_vocab_file = 'uci_vocab.txt'
all_st_notation_vocab_file = 'all_st_notation_vocab.txt'
comments_st_notation_vocab_file = 'comments_st_notation_vocab.txt'

# Check if the files exist
if os.path.exists(fen_vocab_file) and os.path.exists(uci_vocab_file):

    # Load the variables from files
    with open(fen_vocab_file, 'r') as f:
        fen_vocab = f.read().splitlines()
    with open(uci_vocab_file, 'r') as f:
        uci_vocab = f.read().splitlines()
    with open(all_st_notation_vocab_file, 'r') as f:
        all_st_notation_vocab = f.read().splitlines()
    with open(comments_st_notation_vocab_file, 'r') as f:
        comments_st_notation_vocab = f.read().splitlines()

else:

    # Initialize the variables
    fen_vocab = get_FEN_vocab()
    uci_vocab = get_uci_vocab()
    all_st_notation_vocab = get_st_notation_vocab()
    comments_st_notation_vocab = get_comments_st_notation_vocab(all_st_notation_vocab)

    # Save the variables to files
    with open(fen_vocab_file, 'w') as f:
        f.write('\n'.join(fen_vocab))
    with open(uci_vocab_file, 'w') as f:
        f.write('\n'.join(uci_vocab))
    with open(all_st_notation_vocab_file, 'w') as f:
        f.write('\n'.join(all_st_notation_vocab))
    with open(comments_st_notation_vocab_file, 'w') as f:
        f.write('\n'.join(comments_st_notation_vocab))

# Ajout des caractères du vocabulaire FEN à l'objet tokenizer
tokenizer.add_tokens(fen_vocab)
# Ajout des caractères du vocabulaire uci à l'objet tokenizer
tokenizer.add_tokens(uci_vocab)
# Ajout des caractères du vocabulaire des commentaires à l'objet tokenizer
tokenizer.add_tokens(all_st_notation_vocab)

# Save the updated tokenizer in the working directory
tokenizer.save_pretrained(os.getcwd())

# Adaptation de la taille des embeddings avec la taille du nouveau vocabulaire
model.resize_token_embeddings(len(tokenizer))
print("Updated Vocabulary Size:", len(tokenizer))
print("Tous les vocabs sont prets")



# Importations pour la création du train_loader comment et uci
from model_test_2 import get_X_and_y_encoded_comment, extract_tensors, get_X_train_X_test_dataset_comment
from model_test_3 import get_X_and_y_encoded_uci, extract_tensors, get_X_train_X_test_dataset_uci

# Get the training and testing data loaders for comment generation
train_loader_comment, test_loader_comment = get_X_train_X_test_dataset_comment()
# Get the training and testing data loaders for uci prediction
train_loader_uci, test_loader_uci = get_X_train_X_test_dataset_uci()



# Fonction d'entrainement de BART avec un approche Multi-Task
def train_BART_model_multitask(train_loader_comment, train_loader_uci, model, device, num_epochs=5, learning_rate=2e-5):

    # Send the model to the device
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Define the loss functions
    criterion_comment = nn.CrossEntropyLoss()
    criterion_uci = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss_comment = 0
        total_loss_uci = 0
        
        # Training on comments
        for batch_comment, batch_uci in zip(train_loader_comment, train_loader_uci):
            # Move batch to device
            batch_comment = [item.to(device) for item in batch_comment]
            batch_uci = [item.to(device) for item in batch_uci]
            
            # Unpack batch for comments
            input_ids_fens_c, attention_mask_fens_c, input_ids_comments, attention_mask_comments = batch_comment
            # Unpack batch for UCI predictions
            input_ids_fens_u, attention_mask_fens_u, input_ids_uci, attention_mask_uci = batch_uci

            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass for comments
            outputs_comment = model(input_ids=input_ids_fens_c, attention_mask=attention_mask_fens_c, decoder_input_ids=input_ids_comments, decoder_attention_mask=attention_mask_comments)
            logits_comment = outputs_comment.logits
            loss_comment = criterion_comment(logits_comment.view(-1, logits_comment.shape[-1]), input_ids_comments.view(-1))
            total_loss_comment += loss_comment.item()

            # Forward pass for UCI predictions
            outputs_uci = model(input_ids=input_ids_fens_u, attention_mask=attention_mask_fens_u, decoder_input_ids=input_ids_uci, decoder_attention_mask=attention_mask_uci)
            logits_uci = outputs_uci.logits
            loss_uci = criterion_uci(logits_uci.view(-1, logits_uci.shape[-1]), input_ids_uci.view(-1))
            total_loss_uci += loss_uci.item()

            # Backward pass
            loss = loss_comment + loss_uci
            loss.backward()
            # Update weights
            optimizer.step()
            print("Entrainement du batch fini")

            del batch_comment, batch_uci, outputs_comment, outputs_uci, loss_comment, loss_uci, loss

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss (Comments): {total_loss_comment/len(train_loader_comment):.4f}, Loss (UCI): {total_loss_uci/len(train_loader_uci):.4f}')

    print('Training finished!')

    model_path = os.getcwd() + '/model_BART_2.pt'
    model.save_pretrained(model_path)

    print("Model saved to", model_path)

    return model_path


train_BART_model_multitask(train_loader_comment, train_loader_uci, model, device)



def generate_comment_and_move(fen_notation, model, tokenizer, device):

    # Tokenize the input FEN notation
    input_ids = tokenizer.encode(fen_notation, return_tensors='pt').to(device)

    # Generate comment
    generated_comment = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True).to(device)
    decoded_comment = tokenizer.decode(generated_comment[0], skip_special_tokens=True)

    # Generate move (UCI notation)
    generated_move = model.generate(input_ids, max_length=2, num_beams=1, early_stopping=True).to(device)
    decoded_move = tokenizer.decode(generated_move[0], skip_special_tokens=True)

    return decoded_comment, decoded_move