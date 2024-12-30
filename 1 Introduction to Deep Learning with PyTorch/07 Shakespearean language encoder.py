# Define your Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Complete the encoding function
def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer
    
# Complete the text processing pipeline
def text_processing_pipeline(sentences):
    processed_sentences = preprocess_sentences(sentences)
    encoded_sentences, vectorizer = encode_sentences(processed_sentences)
    dataset = ShakespeareDataset(encoded_sentences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer

dataloader, vectorizer = text_processing_pipeline(processed_shakespeare)

# Print the vectorizer's feature names and the first 10 components of the first item
print(vectorizer.get_feature_names_out()[:10]) 
print(next(iter(dataloader))[0, :10])
