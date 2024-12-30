# Complete the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)       
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# Initialize the model
gru_model = GRUModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gru_model.parameters(), lr=0.01)

# Train the model and backpropagate the loss after initialization
for epoch in range(15): 
    optimizer.zero_grad()
    outputs = gru_model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
