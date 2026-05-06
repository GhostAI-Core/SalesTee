import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_translation():
    repo = "stefan7/pocket_polyglot_mzansi_50M_6langs"
    print(f"Loading {repo}...")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    print("Model loaded.")

    text = "Hello, how are you today?"
    src_lang = "eng_Latn"
    tgt_lang = "zul_Latn"

    print(f"Translating: '{text}' ({src_lang} -> {tgt_lang})")
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"Result: {result}")

if __name__ == "__main__":
    test_translation()
