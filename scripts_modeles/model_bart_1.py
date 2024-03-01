# Téléchargement du modèle BART et de son tokeniseur
from download_BART_model import download_model, download_tokenizer
model = download_model()
tokenizer = download_tokenizer()

# Importation de la fonction d'encodage FEN
from pre_processing import encode_fen

# Chargement vocabulaire FEN
fen_vocab_file = 'fen_vocab.txt'
with open(fen_vocab_file, 'r') as f:
    fen_vocab = f.read().splitlines()

# Exemple de position FEN
input_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Fonction pour générer un commentaire à partir d'une position FEN
def comment_generation_model_test_1(input_fen):

    # Encodage de l'entrée
    encoded_input = encode_fen(input_fen, fen_vocab)

    # Utilise le modèle BART pour générer le commentaire
    output_ids = model.generate(encoded_input["input_ids"].unsqueeze(0), max_length=100, num_beams=4, early_stopping=True)

    # Décode la sortie générée
    generated_comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_comment