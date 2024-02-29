# from pre_processing import encode_fen
import os

# Chargement du modèle fine-tuné
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("Mathou/Mbart_chess_comment")
tokenizer = MBart50TokenizerFast.from_pretrained("Mathou/Mbart_chess_comment")

# fen_vocab_file = '../fen_vocab.txt'
# all_st_notation_vocab_file = '../all_st_notation_vocab.txt'
# comments_st_notation_vocab_file = '../comments_st_notation_vocab.txt'

# # # Vérifie si les fichiers existent
# # if os.path.exists(fen_vocab_file) and os.path.exists(all_st_notation_vocab_file) and os.path.exists(comments_st_notation_vocab_file):
# #     # Charge les variables à partir des fichiers
# with open(fen_vocab_file, 'r') as f:
#     fen_vocab = f.read().splitlines()
# with open(all_st_notation_vocab_file, 'r') as f:
#     all_st_notation_vocab = f.read().splitlines()
# with open(comments_st_notation_vocab_file, 'r') as f:
#     comments_st_notation_vocab = f.read().splitlines()

# # Ajoute les caractères du vocabulaire FEN à l'objet tokenizer
# tokenizer.add_tokens(fen_vocab)
# # Ajoute les caractères du vocabulaire des commentaires à l'objet tokenizer
# tokenizer.add_tokens(all_st_notation_vocab)
# # Sauvegarde le tokenizer mis à jour dans le répertoire de travail
# tokenizer.save_pretrained(os.getcwd())
# # Ajuste la taille des embeddings pour correspondre à la taille du nouveau vocabulaire
# model.resize_token_embeddings(len(tokenizer))

def encode_fen(input_fen):

    # On tokenize la notation au niveau du caractère
    tokenized_fen = [car for car in input_fen]

    # On encode la notation FEN avec le tokenizer
    encoded_fen = tokenizer.encode_plus(tokenized_fen, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

    # Formattage pour s'adapter à l'entrée du modèle dans le batch
    encoded_fen = {key: value.squeeze(0) for key, value in encoded_fen.items()}

    # La sortie est une séquence d'entiers en tensor
    return encoded_fen

def comment_generation(fen_input):
    print("fen_input : ", fen_input)
    # Encode l'entrée FEN
    input_tensor = encode_fen(fen_input)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_tensor["input_ids"].unsqueeze(0)
    attention_mask = input_tensor["attention_mask"].unsqueeze(0)

    tokenizer.src_lang = "Zh_CN"
    generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    comment = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return comment[0]
