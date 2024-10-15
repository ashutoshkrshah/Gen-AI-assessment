from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, src_language, dest_language):
    model_name = f'Helsinki-NLP/opus-mt-{src_language}-{dest_language}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example usage
text_to_translate = "Hello, how are you?"
source_language = "en"
destination_language = "es"

translated_text = translate_text(text_to_translate, source_language, destination_language)
print(translated_text)
