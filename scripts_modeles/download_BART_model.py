# Importations
from transformers import BartTokenizer, BartForConditionalGeneration, MBartForConditionalGeneration, MBart50TokenizerFast

# Téléchargement du modèle BART
def download_model():
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    return bart_model

# Téléchargement du tokenizer BART
def download_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    return tokenizer

def download_mbart_model():
    mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt") 
    return mbart_model

def download_mbart_tokenizer():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="zh_CN", tgt_lang="en_XX")
    return tokenizer