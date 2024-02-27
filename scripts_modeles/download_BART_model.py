# Importations
from transformers import BartTokenizer, BartForConditionalGeneration

# Téléchargement du modèle BART
def download_model():
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    return bart_model

# Téléchargement du tokenizer BART
def download_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    return tokenizer