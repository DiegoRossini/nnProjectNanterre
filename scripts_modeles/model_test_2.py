# On utilise la GPU quand c'est possible
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device)
print("Nom de la GPU:", gpu_name)
print(torch.__version__)

if torch.cuda.is_available():
    print("GPU disponibile")
else:
    print("GPU non disponibile")


# Téléchargements des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder, encode_fen, get_FEN_vocab, encode_comment, get_st_notation_vocab, get_comments_st_notation_vocab, tokenize_comment
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import pandas as pd
import time
from sklearn.model_selection import train_test_split



from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

# Initializing a BART facebook/bart-large style configuration
configuration = BartConfig(vocab_size=53291)

# Initializing a model (with random weights) from the facebook/bart-large style configuration
model = BartForConditionalGeneration(configuration)

# Accessing the model configuration
configuration = model.config

# Initialisation du BART Tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Détermine le chemin du corpus
corpus_path = find_corpus_folder(directory='corpus_csv')

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


print("Tous les vocabs sont prets")
print("i'm heeeereeee")



# Fonction d'extraction des fens et des commentaires encodés
def get_X_and_y_encoded():

    # Constitution de X et y vides
    X = []
    y = []

    # Parcours tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # On commence le calcul temporel d'une boucle
        start_time = time.time()

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        # Parcours les lignes du DataFrame et encodage des données
        for fen, comment in zip(df['FEN_notation'], df['Comment']):
            try:
                tokenized_comment = tokenize_comment(comment, comments_st_notation_vocab)
                y.append(encode_comment(tokenized_comment))
                X.append(encode_fen(fen, fen_vocab))
            except:
                pass
        
        # On affiche le temps de traitement de la boucle en question
        end_time = time.time()
        print(end_time - start_time)

    # Division des données en données d'entraînement et de test
    train_encodings_fens, test_encodings_fens, train_encoding_comments, test_encoding_comments = train_test_split(X, y, test_size=0.3, random_state=42)

    # Nettoyage des données
    X = None
    y = None

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
    
    # Nettoyage des données d'entraînement et de test
    train_encodings_fens = None
    test_encodings_fens = None
    train_encoding_comments = None
    test_encoding_comments = None

    # Création objets DataLoader pour les données d'entraînement et de test
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Nettoyage des données dataset
    train_dataset = None
    test_dataset = None

    # Output train_loader, test_loader
    return train_loader, test_loader




def train_BART_model(train_loader, model, device, num_epochs=5, learning_rate=2e-5):

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
            print("Entrainement du batch fini")

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

    print('Training finished!')

    model_path = os.getcwd() + '/model_BART_2.pt'
    model.save_pretrained(model_path)

    print("Model saved to", model_path)

    return model_path



def evaluate_BART_model(test_loader, model, device):

    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and labels
    all_predictions = []
    all_labels = []
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():

        # Boucle pour chaque batch
        for batch in test_loader:

            # Move batch to device
            batch = [item.to(device) for item in batch]

            # Unpack batch
            input_ids_fens, attention_mask_fens, input_ids_comments, attention_mask_comments = batch
            
            # Generate predictions
            outputs = model.generate(input_ids=input_ids_fens, attention_mask=attention_mask_fens)
            predictions = outputs
            
            # Extend predictions and labels lists
            all_predictions.extend(predictions)
            all_labels.extend(input_ids_comments)
    
    # Calculate accuracy
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    accuracy = np.mean(all_predictions == all_labels)
    
    print(f'Accuracy: {accuracy:.4f}')
    
    return accuracy



# # Get the training and testing data loaders
train_loader, test_loader = get_X_train_X_test_dataset()

# Train the model
train_BART_model(train_loader, model, device)



def comment_generation_model_test_2(model_path, fen_input, tokenizer, encode_fen):
    try:
        # Load the model
        model = BartForConditionalGeneration.from_pretrained(model_path)

        # Encode the FEN input
        input_tensor = encode_fen(fen_input)

        # Transfer input tensor to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        
        # Move the model to GPU if available
        model = model.to(device)

        # Use the BART model to generate the comment
        output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

        # Decode the generated output
        generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_comment
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None