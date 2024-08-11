

#%%writefile app.py
import streamlit as st
from bs4 import BeautifulSoup
from io import BytesIO
import PyPDF2
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import requests
import fitz  # PyMuPDF
import random
import time
import io
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
#from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
 
st.title("PDF Classifier")
model_choice = st.selectbox('Select Model', ['Machine Learning', 'Deep Learning'])
 
class DataPreprocess:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def display(self):
        print(self.df.head())
        print()
        print(self.df.info())
 
    # This function gives processed dataframe with correct urls
    def pre_processed(self):
        no_dup = self.df.drop_duplicates()
        corrected_df = no_dup[no_dup['datasheet_link'] != '-']
        corrected_df = corrected_df.dropna()

        # corrected_df = corrected_df[~corrected_df.apply(lambda row: row.astype(str).str.contains('8e9daddd-82b0-4ed4-a656-a8aa011ea6d3').any(), axis=1)]
        corrected_df = corrected_df[corrected_df['datasheet_link'].str.startswith('http')]

        # corrected_df = corrected_df[corrected_df['datasheet_link'].str.endswith('.pdf')]
        corrected_df = corrected_df.reset_index(drop=True)

        return corrected_df

    def incorrect_url(self):
        incorrect_url_df = self.df[~self.df['datasheet_link'].str.startswith('http')]
        incorrect_url_df['datasheet_link'] = 'https:' + incorrect_url_df['datasheet_link']
        return incorrect_url_df

    def processed(self):
        df1 = self.pre_processed()
        df2 = self.incorrect_url()
        return pd.concat([df1, df2], ignore_index=True)


class ExtractData:
    def __init__(self, image_dir='images', max_retries=5):
        self.image_dir = image_dir
        self.max_retries = max_retries
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    def url_process(self, url): # Extracts text from the PDF
        for attempt in range(self.max_retries):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=20)
                time.sleep(1)

                if response.status_code == 200:
                    pdf_file = io.BytesIO(response.content)
                    doc = fitz.open(stream=pdf_file, filetype='pdf')
                    text = ""

                    # Extract text from the first page
                    for page_num in range(min(1, len(doc))):
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        text += page_text

                    if text.strip():
                        return text, []

                    # Extract images if text extraction fails
                    image_paths = self.extract_images(doc)
                    return text, image_paths
            except Exception as e:
                # print(url)
                print(f"Error extracting text from PDF: {e}")

            sleep_time = 2 * attempt + random.uniform(0, 1)
            time.sleep(sleep_time)

        return None, []

    def image_process(self, image_path): # Extracts text from image

        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            # print(f"Error extracting text from image: {e}")
            return None

    def extract_images(self, doc):  # Extract images and returns image paths

        image_paths = []
        for page_num in range(min(1, len(doc))):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{self.image_dir}/page_{page_num}_img_{img_index}.{image_ext}"
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                image_paths.append(image_filename)
        return image_paths

    def process_row(self, row):  # process each url from dataframe to extract text.
        url = row['datasheet_link']
        text, image_paths = self.url_process(url)
        if not text:  # If no text was extracted from the PDF, try OCR on images
            text = ""
            for img_path in image_paths:
                img_text = self.image_process(img_path)
                if img_text:
                    text += " " + img_text
        return text.strip()

    def extract_text(self, df): # uses multiprocessing to extract text from url

        df['content'] = ""   # Text will be stored in new column

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_row, row): idx for idx, row in df.iterrows()}
        # with ProcessPoolExecutor() as executor:
            # Create futures
            # futures = {executor.submit(self.process_row, row): idx for idx, row in df.iterrows()}
            for future in (as_completed(futures)):
                idx = futures[future]
                try:
                    df.at[idx, 'content'] = future.result()
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    df.at[idx, 'content'] = ""

        return df

# DL model
class PDFTextDataset(Dataset):
    def __init__(self, filepath):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = pd.read_csv(filepath)

        self.df['content'] = self.df['content'].fillna('')
        self.df['embedding'] = self.df['content'].apply(lambda x: self.model.encode(x))

        embeddings = self.df['embedding'].tolist()
        embedding_size = len(embeddings[0]) if embeddings else 0
        self.X = torch.tensor(embeddings, dtype=torch.float32)

        self.test_data = TensorDataset(self.X)

        save_path = "/content/drive/MyDrive/Parspec/test_embedding_streamlit.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)

    def get_test_dataloader(self, batch_size=32):
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        return test_loader

class TextClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=4, dropout_rate=0.2):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)


def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load("/content/drive/MyDrive/Parspec/model_data.pth"))

    model.eval()
    all_preds = []

    label_mapping = {2: "lighting", 1: "fuses", 0: "cable", 3: "others"}

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0]  # Extract features
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())

            for pred in predicted.numpy():
                print(f"Prediction: {pred}")

    all_preds = np.array(all_preds)
    all_preds_mapped = [label_mapping[pred] for pred in all_preds]
    return all_preds_mapped


# ML model
def MLdataset(filepath):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        df = pd.read_csv(filepath)

        df['content'] = df['content'].fillna('')
        df['embedding'] = df['content'].apply(lambda x: model.encode(x))

        embeddings = df['embedding'].tolist()

        X_test_np = np.array(embeddings)
        return X_test_np

def evaluate_ML(X):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('/content/drive/MyDrive/Parspec/train_data/xgb_model.json')
    y_pred = xgb_model.predict(X)

    label_mapping = {2: "lighting", 1: "fuses", 0: "cable", 3: "others"}
    y_pred_mapped = [label_mapping[pred] for pred in y_pred]
    return y_pred_mapped

def classifypdf(csv_file):

    csv_path = "uploaded_file.csv"
    with open(csv_path, "wb") as f:
      f.write(csv_file.getbuffer())

    data_processor = DataPreprocess(csv_path)
    preprocessed_data = data_processor.processed()

    extractor = ExtractData()
    df = extractor.extract_text(preprocessed_data)
    df.to_csv('final_df.csv')

    if model_choice == 'Machine Learning':
      dataset = MLdataset('final_df.csv')
      results = evaluate_ML(dataset)
      return results

    elif model_choice == 'Deep Learning':
      dataset = PDFTextDataset('final_df.csv')
      test_loader = dataset.get_test_dataloader()
      model2 = TextClassifier()
      results = evaluate_model(model2, test_loader)
      return results


def main():
  uploaded_file = st.file_uploader("Upload a CSV file containing PDF URLS", type = 'csv')
  if uploaded_file is not None:
      st.write("Processing the file")
      results = classifypdf(uploaded_file)
      st.write("Classification results: ")
      st.write(results)


if __name__ == "__main__":
     main()

