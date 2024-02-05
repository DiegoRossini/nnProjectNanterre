# Importations

import re
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_lg")


# Chargements de la csv de test

df = pd.read_csv("./df_test.csv")


# Extraction d'un vocabulaire des mots composants les commentaires.

comments = df["Comment"]
fens = df["FEN_notation"]
pgns = df["Standard_notation"]

def get_comments_vocab(comment_list, ucis_list, fens_list):

    # Création d'un vocabulaire vide
    vocab = []

    # Ajout des tous les mouvements en notation standard (ils representent, en soi, des mots)
    vocab.extend(ucis_list)

    # Filtre des mouvements déjà notès dans ucis_list et tokenisation des commentaires.
    regex = r'\b(' + '|'.join(ucis_list) + r')\b'
    for comment in comments:
        cleaned_comment = re.sub(regex, "", comment)
        doc = nlp(cleaned_comment)
        for token in doc:
            if token not in vocab:            
                vocab.append(token.text)

    return vocab
        

custom_vocab = get_comments_vocab(comments, pgns, fens)


        

# Tentatif de fine-tuner le BertTokenizer sur le vocabulaire de l'article de Git 2018.

from transformers import BertTokenizer, BertConfig

# Spécification du vocabulaire spécialisé pour fine-tuner le tokeniseur de BERT
custom_vocab.extend(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

# Céation d'une configuration specialisée pour BERT
config = BertConfig(vocab_size=len(custom_vocab))

# Chargement du tokeniseur pré-entrainé de BERT
pretrained_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Adaptation du tokeniseur de BERT pour le vocabulaire spécifié.
pretrained_tokenizer.add_tokens(custom_vocab, special_tokens=True)
pretrained_tokenizer.save_pretrained('fine_tuned_BERT_tokenizer')

test = pretrained_tokenizer.tokenize("just striving for my usual setup in the Hungarion variation in the Sicilian: Ne7, ..., 0-0, ..., and play or prepare d5... best for White is probably Nc3, or exchanging the center with a direct d4...")

print(test)