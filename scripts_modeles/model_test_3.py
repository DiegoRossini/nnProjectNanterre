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


# Importations depuis pre_processing.py
from pre_processing import find_corpus_folder, get_FEN_vocab, encode_fen, get_uci_vocab, encode_uci
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
uci_vocab_file = 'uci_vocab.txt'

# Check if the files exist
if os.path.exists(fen_vocab_file) and os.path.exists(uci_vocab_file):
    # Load the variables from files
    with open(fen_vocab_file, 'r') as f:
        fen_vocab = f.read().splitlines()
    with open(uci_vocab_file, 'r') as f:
        uci_vocab = f.read().splitlines()
else:
    # Initialize the variables
    fen_vocab = get_FEN_vocab()
    uci_vocab = get_uci_vocab()

    # Save the variables to files
    with open(fen_vocab_file, 'w') as f:
        f.write('\n'.join(fen_vocab))
    with open(uci_vocab_file, 'w') as f:
        f.write('\n'.join(uci_vocab))

# Ajout des caractères du vocabulaire FEN à l'objet tokenizer
tokenizer.add_tokens(fen_vocab)
# Ajout des caractères du vocabulaire uci à l'objet tokenizer
tokenizer.add_tokens(uci_vocab)
# Save the updated tokenizer in the working directory
tokenizer.save_pretrained(os.getcwd())
# Get the vocabulary size from the tokenizer
vocab_size = tokenizer.vocab_size
print("Updated Vocabulary Size:", vocab_size)

print("Tous les vocabs sont prets")



# Fonction d'extraction des fens et des uci encodés
def get_X_and_y_encoded():

    # Constitution de X et y vides
    X = []
    y = []

    x = 0
    # Parcours tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        if x != 101:

            # On commence le calcul temporel d'une boucle
            start_time = time.time()

            # Charge le fichier CSV dans un DataFrame pandas
            df = pd.read_csv(csv_match_path)

            # Boucle sur chaque paire de valeurs UCI_notation et N_move dans le DataFrame df
            for idx, (uci_moves) in enumerate(zip(df['UCI_notation'], df['N_move'])):
                
                # Tentative d'exécution du bloc suivant
                try:

                    # Vérification si le mouvement est le premier de la partie
                    if uci_moves[1] == 1:

                        # Si oui, définir la position initiale FEN standard
                        start_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

                        # Encodage de la position FEN et de l'action UCI correspondante
                        X.append(encode_fen(start_fen, fen_vocab))
                        y.append(encode_uci(uci_moves[0], uci_vocab))
                    
                    # Si ce n'est pas le premier mouvement
                    else:

                        # On récupérère la FEN précédente à partir du DataFrame à l'indice -1
                        previous_fen = df.at[df.index[idx -1], 'FEN_notation']

                        # Encodage de la FEN précédente et de l'action UCI correspondante
                        X.append(encode_fen(previous_fen, fen_vocab))
                        y.append(encode_uci(uci_moves[0], uci_vocab))

                # En cas d'erreur, passer à l'itération suivante sans rien faire
                except:
                    pass
        
            # On affiche le temps de traitement de la boucle en question
            end_time = time.time()
            print(end_time - start_time)

            x += 1

        else:
            break
        
    # Division des données en données d'entraînement et de test
    train_fen_encodings, train_uci_encodings, test_fen_encodings, test_uci_encodings = train_test_split(X, y, test_size=0.2, random_state=42)

    # Nettoyage des données
    X = None
    y = None

    # Output X_train, X_test, y_train, y_test
    return train_fen_encodings, test_fen_encodings, train_uci_encodings, test_uci_encodings



# Function to extract tensors from a list of dictionaries
def extract_tensors(data):
    
    input_ids_list = [item["input_ids"] for item in data]
    attention_mask_list = [item["attention_mask"] for item in data]
    
    # Concatenate or stack tensors
    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
    
    return input_ids_tensor, attention_mask_tensor



def get_X_train_X_test_dataset():

    # Obtention des X_train et X_test encodés
    train_fen_encodings, train_uci_encodings, test_fen_encodings, test_uci_encodings = get_X_and_y_encoded()

    # Extract tensors from the training data
    train_input_ids_fens, train_attention_mask_fens = extract_tensors(train_fen_encodings)
    train_input_ids_uci, train_attention_mask_uci = extract_tensors(train_uci_encodings)

    # Extract tensors from the testing data
    test_input_ids_fens, test_attention_mask_fens = extract_tensors(test_fen_encodings)
    test_input_ids_uci, test_attention_mask_uci = extract_tensors(test_uci_encodings)

    print("Train Input IDs Fens Shape:", train_input_ids_fens.shape)
    print("Train Attention Mask Fens Shape:", train_attention_mask_fens.shape)
    print("Train Input IDs UCI Shape:", train_input_ids_uci.shape)
    print("Train Attention Mask UCI Shape:", train_attention_mask_uci.shape)


    # Create TensorDatasets
    train_dataset = TensorDataset(train_input_ids_fens, train_attention_mask_fens,
                                train_input_ids_uci, train_attention_mask_uci)
    test_dataset = TensorDataset(test_input_ids_fens, test_attention_mask_fens,
                                test_input_ids_uci, test_attention_mask_uci)
    
    # Nettoyage des données d'entraînement et de test
    train_fen_encodings = None
    test_fen_encodings = None
    train_uci_encodings = None
    test_uci_encodings = None

    # Création objets DataLoader pour les données d'entraînement et de test
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # Output train_loader, test_loader
    return train_loader, test_loader



def train_BART_model(train_loader, model, device, num_epochs=5, learning_rate=2e-5):

    # Send the model to the device
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

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
        for batch in test_loader:
            # Move batch to device
            batch = [item.to(device) for item in batch]
            # Unpack batch
            input_ids, attention_mask, labels = batch
            
            # Generate predictions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs
            
            # Extend predictions and labels lists
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
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



def predict_next_move(fen_input, model, tokenizer, encode_fen, fen_vocab):
    try:
        # Encode the FEN input
        encoded_fen = encode_fen(fen_input, fen_vocab)

        # Transfer input tensor to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = encoded_fen.to(device)

        # Move the model to GPU if available
        model = model.to(device)

        # Use the model to generate the comment
        output_ids = model.generate(input_tensor.unsqueeze(0), max_length=50, num_beams=4, early_stopping=True)

        # Decode the generated output
        generated_next_move = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_next_move
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None