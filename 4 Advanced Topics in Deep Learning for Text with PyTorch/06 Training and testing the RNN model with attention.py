for epoch in range(epochs):
    attention_model.train()
    optimizer.zero_grad()
    padded_inputs = pad_sequences(inputs)
    outputs = attention_model(padded_inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

for input_seq, target in zip(input_data, target_data):
    input_test = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
   	
    #  Set the RNN model to evaluation mode
    rnn_model.eval()
    # Get the RNN output by passing the appropriate input 
    rnn_output = rnn_model(input_test)
    # Extract the word with the highest prediction score 
    rnn_prediction = ix_to_word[torch.argmax(rnn_output).item()]

    attention_model.eval()
    attention_output = attention_model(input_test)
    # Extract the word with the highest prediction score
    attention_prediction = ix_to_word[torch.argmax(attention_output).item()]

    print(f"\nInput: {' '.join([ix_to_word[ix] for ix in input_seq])}")
    print(f"Target: {ix_to_word[target]}")
    print(f"RNN prediction: {rnn_prediction}")
    print(f"RNN with Attention prediction: {attention_prediction}")
