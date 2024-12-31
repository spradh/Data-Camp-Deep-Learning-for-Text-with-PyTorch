text = "I had an awesome day!"

# Tokenize the text and return PyTorch tensors
input_eval = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=32)
outputs_eval = model(**input_eval)

# Convert the output logits to probabilities
predictions = torch.nn.functional.softmax(outputs_eval.logits, dim=-1)

# Display the sentiments
predicted_label = "Positive" if torch.argmax(predictions) > 0 else "Negative"
print(f"Text: {text}\nSentiment: {predicted_label}")
