# Define the generator class
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, seq_length), 
            nn.Sigmoid())
    def forward(self, x):
        return self.model(x)

# Define the discriminator networks
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, 1), 
            nn.Sigmoid())
    def forward(self, x):
        return self.model(x)
