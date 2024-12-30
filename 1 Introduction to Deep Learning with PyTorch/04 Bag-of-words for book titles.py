# Import from sklearn
from sklearn.feature_extraction.text import CountVectorizer

titles = ['The Great Gatsby','To Kill a Mockingbird','1984','The Catcher in the Rye','The Hobbit', 'Great Expectations']

# Initialize Bag-of-words with the list of book titles
vectorizer = CountVectorizer()
bow_encoded_titles = vectorizer.fit_transform(titles)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(bow_encoded_titles.toarray()[0, :5])
