# Initalize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_prompt = "translate English to French: 'Hello, how are you?'"

# Encode the input prompt using the tokenizer
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

# Generate the translated ouput
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:",generated_text)
