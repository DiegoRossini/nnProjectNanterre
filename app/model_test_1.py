# Téléchargement du modèle BART et de son tokeniseur
from download_BART_model import download_model, download_tokenizer
model = download_model()
tokenizer = download_tokenizer()

# Importa il vocabolario FEN dal file pre_processing.py
from pre_processing import encode_fen

import torch

# Chargement vocabulaire FEN
fen_vocab_file = '../fen_vocab.txt'
with open(fen_vocab_file, 'r') as f:
    fen_vocab = f.read().splitlines()

# Esempio di utilizzo
input_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

encoded_input = encode_fen(input_fen, fen_vocab)



'''
-------------- TENTATIve 1 DE FAIRE TOURNER LE MODELE------------------------
'''



def comment_generation_model_test_1(input_fen):
    """
    Génère un commentaire en fonction de l'entrée encodée en utilisant le modèle BART.
    
    Args:
        encoded_input (list): La séquence d'entrée encodée.
        model: Le modèle BART pour la génération de commentaires.
        tokenizer: Le tokenizer pour le modèle BART.
    
    Returns:
        str: Le commentaire généré.
    """
    
    encoded_input = encode_fen(input_fen, fen_vocab)

    # Utilise le modèle BART pour générer le commentaire
    output_ids = model.generate(encoded_input["input_ids"].unsqueeze(0), max_length=20, num_beams=4, early_stopping=True)

    # Décode la sortie générée
    generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_comment

# Exemple d'utilisation :
# generated_comment = comment_generation_model_test_1(encoded_input, model, tokenizer)
# print("Commentaire généré :", generated_comment)



# # Input
# input_text = "The movie was really good. I liked the plot and the acting was great. The special effects were amazing."

# # Codifica l'input
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# # Generazione del testo
# output = model.generate(input_ids, max_length=500, num_beams=10, early_stopping=True)

# # Decodifica l'output
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print("Output generato:", output_text)