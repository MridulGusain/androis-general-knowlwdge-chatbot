import pandas as pd
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset

file_path = 'C:/Users/acer/OneDrive/Documents/chatbott/android_chatbot_dataset_with_tags.csv'
data = pd.read_csv(file_path)

# Extract patterns, tags, and answers
patterns = data['patterns'].tolist()
tags = data['tags'].tolist()

# Encode the tags into numeric labels
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(patterns, encoded_tags, test_size=0.2, random_state=42)

# Text preprocessing function
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])


# Preprocess the training and testing data
X_train = [preprocess_text(text) for text in X_train]
X_test = [preprocess_text(text) for text in X_test]

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate model
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.2f}')

# Save components for later use
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
