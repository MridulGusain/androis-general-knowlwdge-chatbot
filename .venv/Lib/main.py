import pandas as pd
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import scrolledtext

#Load saved components
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the dataset for responses
file_path = 'C:/Users/acer/OneDrive/Documents/chatbott/android_chatbot_dataset_with_tags.csv'
data = pd.read_csv(file_path)


#Preprocessing function
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])

#Function to handle user input
# Function to handle user input
# Function to handle user input
def send():
    user_input = entry.get()
    if user_input.strip() == "":  # Check for empty input
        return

    # Insert the user's input into the chat window
    chat_window.config(state=tk.NORMAL)  # Enable editing
    chat_window.insert(tk.END, "You: " + user_input + "\n")

    if user_input.lower() == "exit":
        chat_window.insert(tk.END, "Chatbot: Goodbye!\n")
        chat_window.config(state=tk.DISABLED)  # Disable editing
        root.quit()
        return

    # Process and predict
    user_input_processed = preprocess_text(user_input)
    user_input_vectorized = vectorizer.transform([user_input_processed])
    predicted_tag = model.predict(user_input_vectorized)[0]
    tag = label_encoder.inverse_transform([predicted_tag])[0]

    # Select a response from the dataset
    response = random.choice(data[data['tags'] == tag]['answers'].tolist())

    chat_window.insert(tk.END, "Chatbot: " + response + "\n")
    chat_window.config(state=tk.DISABLED)

    chat_window.see(tk.END)

    entry.delete(0, tk.END)


#Create the main window
root = tk.Tk()
root.title("Android General Knowledge Chatbot")

# Create a scrolled text area for chat history
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_window.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_window.config(state=tk.DISABLED)  # Make it read-only

# Create an entry field for user input
entry = tk.Entry(root, width=80)
entry.pack(pady=10, padx=10)

# Create a send button
send_button = tk.Button(root, text="Send", command=send)
send_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
