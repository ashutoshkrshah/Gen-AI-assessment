# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:36:48 2024

@author: DELL
"""

## Write a python program to generate creative writing by giving a starting sentence as a prompt using GPT-2.
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

# Define the starting sentence
sentence = 'I love cats'

# Convert the sentence into token IDs
numeric_ids = tokenizer.encode(sentence, return_tensors='pt')

# Generate text given the sentence
result = model.generate(
    numeric_ids,
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Decode the generated token IDs back into text
generated_text = tokenizer.decode(result[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)