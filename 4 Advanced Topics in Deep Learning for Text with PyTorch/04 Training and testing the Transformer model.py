for epoch in range(5):  
    for sentence, label in zip(train_sentences, train_labels):
        # Split the sentences into tokens and stack the embeddings
        tokens = sentence.split()
        data = torch.stack([token_embeddings[token] for token in tokens], dim=1)
        output = model(data)
        loss = criterion(output, torch.tensor([label]))
        # Zero the gradients and perform a backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

def predict(sentence):
    model.eval()
    # Deactivate the gradient computations and get the sentiment prediction.
    with torch.no_grad():
        tokens = sentence.split()
        data = torch.stack([token_embeddings.get(token, torch.rand((1, 512))) for token in tokens], dim=1)
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        return "Positive" if predicted.item() == 1 else "Negative"

sample_sentence = "This product can be better"
print(f"'{sample_sentence}' is {predict(sample_sentence)}")
