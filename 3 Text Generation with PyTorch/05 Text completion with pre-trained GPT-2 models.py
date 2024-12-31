# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize the pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

seed_text = "Once upon a time"

# Encode the seed text to get input tensors
input_ids = tokenizer.encode(seed_text, return_tensors='pt')

# Generate text from the model
output = model.generate(input_ids, max_length=100, temperature=0.7, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id) 

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
