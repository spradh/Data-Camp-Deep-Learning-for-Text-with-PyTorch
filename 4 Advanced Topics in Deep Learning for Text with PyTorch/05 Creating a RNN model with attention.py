class RNNWithAttentionModel(nn.Module):
    def __init__(self):
        super(RNNWithAttentionModel, self).__init__()
        # Create an embedding layer for the vocabulary
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # Apply a linear transformation to get the attention scores
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        #  Get the attention weights
        attn_weights = torch.nn.functional.softmax(self.attention(out).squeeze(2), dim=1)
        # Compute the context vector 
        context = torch.sum(attn_weights.unsqueeze(2) * out, dim=1)
        out = self.fc(context)
        return out
      
attention_model = RNNWithAttentionModel()
optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
print("Model Instantiated")
