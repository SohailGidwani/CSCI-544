"""
CSCI 544 - Homework Assignment 1
Sentiment Analysis using Amazon Reviews
Name : Sohail Haresh Gidwani
USC ID : 7321203258

Python Version: 3.x
"""

import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ============================================================
# 1. Dataset Preparation
# ============================================================

# Read the data
df = pd.read_csv('data.tsv', sep='\t', on_bad_lines='skip')

# Keep only review_body and star_rating
df = df[['review_body', 'star_rating']].copy()
df = df.dropna()
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
df = df.dropna()
df['star_rating'] = df['star_rating'].astype(int)

# Print three sample reviews with ratings
print("Three Sample Reviews with Ratings:")
for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"Rating: {df['star_rating'].iloc[i]} stars")
    print(f"Review: {df['review_body'].iloc[i]}")
print()

# Print rating statistics
print("Rating Statistics:")
rating_counts = df['star_rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    print(f"{rating}-star reviews: {count}")
print()

# Create three classes
positive_df = df[df['star_rating'] > 3].copy()
negative_df = df[df['star_rating'] <= 2].copy()
neutral_df = df[df['star_rating'] == 3].copy()

# Print class statistics
print(f"Positive reviews: {len(positive_df)}")
print(f"Negative reviews: {len(negative_df)}")
print(f"Neutral reviews: {len(neutral_df)}")

# Sample 100,000 from each class
positive_sample = positive_df.sample(n=100000, random_state=42)
negative_sample = negative_df.sample(n=100000, random_state=42)

# Assign binary labels
positive_sample['label'] = 1
negative_sample['label'] = 0

# Combine and shuffle
df_balanced = pd.concat([positive_sample, negative_sample], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================
# 2. Data Cleaning
# ============================================================

# Contractions dictionary
contractions_dict = {
    "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
    "let's": "let us", "it's": "it is", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "doesn't": "does not", "don't": "do not", "didn't": "did not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "mightn't": "might not", "mustn't": "must not", "needn't": "need not",
    "shan't": "shall not", "wasn't": "was not", "weren't": "were not",
    "what's": "what is", "who's": "who is", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what'll": "what will", "who'll": "who will", "where'll": "where will",
    "what've": "what have", "who've": "who have", "could've": "could have",
    "would've": "would have", "should've": "should have", "might've": "might have",
    "must've": "must have", "y'all": "you all", "ain't": "is not",
    "gonna": "going to", "gotta": "got to", "wanna": "want to",
    "kinda": "kind of", "sorta": "sort of", "lemme": "let me", "gimme": "give me"
}

contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                   flags=re.IGNORECASE)

def expand_contractions(text):
    def replace(match):
        return contractions_dict.get(match.group(0).lower(), match.group(0))
    return contractions_pattern.sub(replace, text)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Calculate average length before cleaning
avg_length_before_cleaning = df_balanced['review_body'].astype(str).apply(len).mean()

# Apply cleaning
df_balanced['cleaned_review'] = df_balanced['review_body'].apply(clean_text)

# Calculate average length after cleaning
avg_length_after_cleaning = df_balanced['cleaned_review'].apply(len).mean()

print(f"Average length before cleaning: {avg_length_before_cleaning:.4f}")
print(f"Average length after cleaning: {avg_length_after_cleaning:.4f}")

# ============================================================
# 3. Preprocessing
# ============================================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Calculate average length before preprocessing
avg_length_before_preprocessing = df_balanced['cleaned_review'].apply(len).mean()

# Apply preprocessing
df_balanced['preprocessed_review'] = df_balanced['cleaned_review'].apply(preprocess_text)

# Calculate average length after preprocessing
avg_length_after_preprocessing = df_balanced['preprocessed_review'].apply(len).mean()

print(f"Average length before preprocessing: {avg_length_before_preprocessing:.4f}")
print(f"Average length after preprocessing: {avg_length_after_preprocessing:.4f}")

# Print three sample reviews before and after data cleaning + preprocessing
print("\nThree Sample Reviews - Before and After Data Cleaning + Preprocessing:")
for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"Original: {df_balanced['review_body'].iloc[i]}")
    print(f"After Cleaning: {df_balanced['cleaned_review'].iloc[i]}")
    print(f"After Preprocessing: {df_balanced['preprocessed_review'].iloc[i]}")

# ============================================================
# 4. Feature Extraction (Bigrams)
# ============================================================

vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(df_balanced['preprocessed_review'])
y = df_balanced['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 5. Perceptron
# ============================================================

perceptron = Perceptron(random_state=42)
perceptron.fit(X_train, y_train)

y_train_pred = perceptron.predict(X_train)
y_test_pred = perceptron.predict(X_test)

print(f"Perceptron Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Perceptron Training Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Perceptron Training Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"Perceptron Training F1-score: {f1_score(y_train, y_train_pred):.4f}")
print(f"Perceptron Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Perceptron Testing Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Perceptron Testing Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Perceptron Testing F1-score: {f1_score(y_test, y_test_pred):.4f}")

# ============================================================
# 6. SVM
# ============================================================

svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)

y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

print(f"SVM Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"SVM Training Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"SVM Training Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"SVM Training F1-score: {f1_score(y_train, y_train_pred):.4f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"SVM Testing Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"SVM Testing Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"SVM Testing F1-score: {f1_score(y_test, y_test_pred):.4f}")

# ============================================================
# 7. Logistic Regression
# ============================================================

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(f"Logistic Regression Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Logistic Regression Training Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Logistic Regression Training Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"Logistic Regression Training F1-score: {f1_score(y_train, y_train_pred):.4f}")
print(f"Logistic Regression Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Logistic Regression Testing Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Logistic Regression Testing Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Logistic Regression Testing F1-score: {f1_score(y_test, y_test_pred):.4f}")

# ============================================================
# 8. Naive Bayes
# ============================================================

nb = MultinomialNB()
nb.fit(X_train, y_train)

y_train_pred = nb.predict(X_train)
y_test_pred = nb.predict(X_test)

print(f"Naive Bayes Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Naive Bayes Training Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Naive Bayes Training Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"Naive Bayes Training F1-score: {f1_score(y_train, y_train_pred):.4f}")
print(f"Naive Bayes Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Naive Bayes Testing Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Naive Bayes Testing Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Naive Bayes Testing F1-score: {f1_score(y_test, y_test_pred):.4f}")
