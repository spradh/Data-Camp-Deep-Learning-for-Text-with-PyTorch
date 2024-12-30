book_reviews = [
    "I love this book".split(),
    "I do not like this book".split()
]
for review in book_reviews:
    # Convert the review words into tensor form
    input_tensor = torch.tensor([word_to_ix[w] for w in review], dtype=torch.long).unsqueeze(0) 
    # Get the model's output
    outputs = model(input_tensor)
    # Find the index of the most likely sentiment category
    _, predicted_label = torch.max(outputs.data, 1)
    # Convert the predicted label into a sentiment string
    sentiment = "Positive" if predicted_label.item()==1 else "Negative"
    print(f"Book Review: {' '.join(review)}")
    print(f"Sentiment: {sentiment}\n")
