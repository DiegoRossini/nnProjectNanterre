# from pre_processing import encode_fen
import os
import torch

# Chargement du modèle fine-tuné
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BartTokenizer

model = MBartForConditionalGeneration.from_pretrained("Mathou/Mbart_chess_comment")
tokenizer = MBart50TokenizerFast.from_pretrained("Mathou/Mbart_chess_comment", src_lang="zh_CN", tgt_lang="en_XX")

def encode_fen(input_fen):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # On tokenize la notation au niveau du caractère
    tokenized_fen = [car for car in input_fen]

    # On encode la notation FEN avec le tokenizer
    encoded_fen = tokenizer.encode_plus(tokenized_fen, return_tensors="pt", padding="max_length", max_length=256, truncation=True)

    # Formattage pour s'adapter à l'entrée du modèle dans le batch
    encoded_fen = {key: value.squeeze(0) for key, value in encoded_fen.items()}

    # La sortie est une séquence d'entiers en tensor
    return encoded_fen



def comment_generation(fen_input):
    # Encode l'entrée FEN
    input_tensor = encode_fen(fen_input)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_tensor["input_ids"].unsqueeze(0)
    attention_mask = input_tensor["attention_mask"].unsqueeze(0)

    tokenizer.src_lang = "Zh_CN"
    generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    comment = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return comment[0]


# input_fen = "5rk1/1p4pp/1q2p3/p2p4/Pn4Q1/1PNP4/2PK1PPP/4R3 b - - 3 22"
# generated_comment = comment_generation(input_fen, model, tokenizer)
# print("Generated Comment:", generated_comment)