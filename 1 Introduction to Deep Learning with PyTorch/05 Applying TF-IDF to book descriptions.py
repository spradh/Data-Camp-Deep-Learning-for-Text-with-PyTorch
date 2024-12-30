# Importing TF-IDF from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF encoding vectorizer
vectorizer = TfidfVectorizer()
tfidf_encoded_descriptions = vectorizer.fit_transform(descriptions)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(tfidf_encoded_descriptions.toarray()[0, :5])
