# Importations
from transformers import BartTokenizer, BartForConditionalGeneration

# Téléchargement du modèle BART
def download_model():
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    return bart_model, tokenizer