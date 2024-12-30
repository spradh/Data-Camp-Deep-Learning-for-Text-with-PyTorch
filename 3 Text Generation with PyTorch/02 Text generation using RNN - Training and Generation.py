# Instantiate the loss function
criterion = nn.CrossEntropyLoss()
# Instantiate the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item()}')

# Test the model
model.eval()
test_input = char_to_ix['r']
test_input = nn.functional.one_hot(torch.tensor(test_input).view(-1, 1), num_classes=len(chars)).float()
predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()
print(f"Test Input: 'r', Predicted Output: '{ix_to_char[predicted_char_ix]}'")
