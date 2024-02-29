from pre_processing import encode_fen

# Chargement du modèle fine-tuné
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("Mathou/Mbart_chess_comment")
tokenizer = MBart50TokenizerFast.from_pretrained("Mathou/Mbart_chess_comment")

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
