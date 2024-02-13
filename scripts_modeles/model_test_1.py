# Téléchargement du modèle BART et de son tokeniseur
from download_BART_model import download_model
model, tokenizer = download_model()


# Importa il vocabolario FEN dal file pre_processing.py
from pre_processing import get_FEN_vocab

import torch


# Ottieni il vocabolario FEN
fen_vocab = get_FEN_vocab()

def encode_fen(input_fen, fen_vocab):
    encoded_input = [fen_vocab.index(char) for char in input_fen]
    encoded_input.append(1)  # Aggiungi <end> codificato
    encoded_input.insert(0, 0)  # Aggiungi <start> codificato
    return encoded_input

# Esempio di utilizzo
input_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

encoded_input = encode_fen(input_fen, fen_vocab)
print("Input codificato:", encoded_input)



'''
-------------- TENTATIF 1 DE FAIRE TOURNER LE MODELE------------------------
'''


def comment_generation_model_test_1(encoded_input, model, tokenizer):
    """
    Génère un commentaire en fonction de l'entrée encodée en utilisant le modèle BART.
    
    Args:
        encoded_input (list): La séquence d'entrée encodée.
        model: Le modèle BART pour la génération de commentaires.
        tokenizer: Le tokenizer pour le modèle BART.
    
    Returns:
        str: Le commentaire généré.
    """
    # Ajoute du padding à la séquence encodée pour uniformiser la longueur de la séquence
    max_length = 100  # Longueur maximale autorisée pour l'entrée
    padded_encoded_input = encoded_input[:max_length] + [0] * (max_length - len(encoded_input))

    # Transforme l'entrée en un tenseur PyTorch
    input_tensor = torch.tensor(padded_encoded_input)

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

# Exemple d'utilisation :
generated_comment = comment_generation_model_test_1(encoded_input, model, tokenizer)
print("Commentaire généré :", generated_comment)