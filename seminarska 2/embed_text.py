import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from bs4 import BeautifulSoup
import requests

df = pd.read_csv("News_Category_Dataset_IS_course_preprocessed.csv")

# Load pre-trained model for both title/description and text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a given link
def extract_text(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Extract text from the links and create embeddings
df['text'] = df['link'].apply(extract_text)

# Create embeddings for title/description and text
title_description_embeddings = df.apply(lambda row: model.encode(row['headline'] + " " + row['short_description']), axis=1)
text_embeddings = df['text'].apply(lambda x: model.encode(x))

# Concatenate title/description and text embeddings
df['final_vector'] = torch.cat([title_description_embeddings.values.tolist(), text_embeddings.values.tolist()], dim=1).tolist()

df.to_csv("sbert_embeddings.csv", header=None)
