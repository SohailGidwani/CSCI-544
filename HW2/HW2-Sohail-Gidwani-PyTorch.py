"""
CSCI 544 - Homework Assignment 2
Sentiment Analysis with Word Embeddings and Neural Networks
Name: Sohail Haresh Gidwani
USC ID: 7321203258

Library: PyTorch
"""

import pandas as pd
import numpy as np
import re
import gc
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import gensim.downloader as api

warnings.filterwarnings('ignore')
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. Dataset Generation
# ============================================================

df = pd.read_csv('amazon_reviews_us_Office_Products_v1_00.tsv', sep='\t', on_bad_lines='skip')
df = df[['review_body', 'star_rating']].copy()
df = df.dropna()
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
df = df.dropna()
df['star_rating'] = df['star_rating'].astype(int)

# Sample 50K per rating for a balanced 250K dataset
balanced_parts = []
for rating in range(1, 6):
    subset = df[df['star_rating'] == rating]
    sampled = subset.sample(n=50000, random_state=SEED)
    balanced_parts.append(sampled)

df_balanced = pd.concat(balanced_parts, ignore_index=True)
del df, balanced_parts
gc.collect()


def assign_sentiment(rating):
    if rating > 3:
        return 1  # positive
    elif rating < 3:
        return 2  # negative
    else:
        return 3  # neutral


df_balanced['sentiment'] = df_balanced['star_rating'].apply(assign_sentiment)

print(f"Balanced dataset size: {len(df_balanced)}")
print(f"  Class 1 (positive): {(df_balanced['sentiment'] == 1).sum()}")
print(f"  Class 2 (negative): {(df_balanced['sentiment'] == 2).sum()}")
print(f"  Class 3 (neutral):  {(df_balanced['sentiment'] == 3).sum()}")

# ============================================================
# Data Cleaning and Preprocessing (same as HW1)
# ============================================================

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
    "what's": "what is", "who's": "who is", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "gonna": "going to", "gotta": "got to", "wanna": "want to",
    "kinda": "kind of", "sorta": "sort of", "lemme": "let me", "gimme": "give me"
}
contractions_pattern = re.compile(
    '({})'.format('|'.join(contractions_dict.keys())), flags=re.IGNORECASE
)


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


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    words = text.split()
    words = [w for w in words if w.lower() not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)


print("Cleaning and preprocessing reviews...")
df_balanced['cleaned'] = df_balanced['review_body'].apply(clean_text)
df_balanced['processed'] = df_balanced['cleaned'].apply(preprocess_text)
df_balanced['tokens'] = df_balanced['processed'].apply(lambda x: x.split())
df_balanced.drop(columns=['review_body', 'star_rating', 'cleaned'], inplace=True, errors='ignore')
gc.collect()
print("Done.")

# Train-test split
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=SEED)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Training set: {len(train_df)}, Testing set: {len(test_df)}")

# ============================================================
# 2. Word Embedding
# ============================================================

# 2(a) Pretrained Word2Vec
print("\nLoading pretrained word2vec-google-news-300...")
pretrained_w2v = api.load('word2vec-google-news-300')
print(f"Pretrained model vocabulary size: {len(pretrained_w2v)}")

# Semantic similarity examples (Pretrained)
result_1 = pretrained_w2v.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
print("\nExample 1 (Pretrained): king - man + woman =")
for word, sim in result_1:
    print(f"  {word}: {sim:.4f}")

sim_score = pretrained_w2v.similarity('excellent', 'outstanding')
print(f"\nExample 2 (Pretrained): similarity('excellent', 'outstanding') = {sim_score:.4f}")

# 2(b) Custom Word2Vec
print("\nTraining custom Word2Vec model...")
all_tokens = df_balanced['tokens'].tolist()
custom_w2v_model = Word2Vec(
    sentences=all_tokens,
    vector_size=300,
    window=11,
    min_count=10,
    seed=SEED,
    workers=1
)
custom_w2v = custom_w2v_model.wv
del custom_w2v_model, all_tokens
gc.collect()
print(f"Custom model vocabulary size: {len(custom_w2v)}")

# Semantic similarity examples (Custom)
try:
    result_1_custom = custom_w2v.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
    print("\nExample 1 (Custom): king - man + woman =")
    for word, sim in result_1_custom:
        print(f"  {word}: {sim:.4f}")
except KeyError as e:
    print(f"\nExample 1 (Custom): Word not in vocabulary - {e}")

try:
    sim_score_custom = custom_w2v.similarity('excellent', 'outstanding')
    print(f"\nExample 2 (Custom): similarity('excellent', 'outstanding') = {sim_score_custom:.4f}")
except KeyError as e:
    print(f"\nExample 2 (Custom): Word not in vocabulary - {e}")


# ============================================================
# Feature Extraction Helpers
# ============================================================

def get_avg_w2v(tokens, w2v_model, dim=300):
    """Compute the average Word2Vec vector for a list of tokens."""
    vectors = [w2v_model[t] for t in tokens if t in w2v_model]
    if len(vectors) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def get_concat_w2v(tokens, w2v_model, num_words=10, dim=300):
    """Concatenate the first num_words Word2Vec vectors."""
    vectors = []
    for token in tokens:
        if len(vectors) >= num_words:
            break
        if token in w2v_model:
            vectors.append(w2v_model[token])
    while len(vectors) < num_words:
        vectors.append(np.zeros(dim))
    return np.concatenate(vectors).astype(np.float32)


# Build average features
print("\nComputing average Word2Vec features...")
X_train_avg_pre = np.array([get_avg_w2v(t, pretrained_w2v) for t in train_df['tokens']])
X_test_avg_pre = np.array([get_avg_w2v(t, pretrained_w2v) for t in test_df['tokens']])
X_train_avg_cus = np.array([get_avg_w2v(t, custom_w2v) for t in train_df['tokens']])
X_test_avg_cus = np.array([get_avg_w2v(t, custom_w2v) for t in test_df['tokens']])

# Labels
y_train_all = train_df['sentiment'].values
y_test_all = test_df['sentiment'].values

# Binary subset (class 1 and class 2 only)
train_binary_mask = np.isin(y_train_all, [1, 2])
test_binary_mask = np.isin(y_test_all, [1, 2])
y_train_bin = y_train_all[train_binary_mask]
y_test_bin = y_test_all[test_binary_mask]

X_train_avg_pre_bin = X_train_avg_pre[train_binary_mask]
X_test_avg_pre_bin = X_test_avg_pre[test_binary_mask]
X_train_avg_cus_bin = X_train_avg_cus[train_binary_mask]
X_test_avg_cus_bin = X_test_avg_cus[test_binary_mask]

print(f"Binary train: {len(y_train_bin)}, test: {len(y_test_bin)}")
print(f"Ternary train: {len(y_train_all)}, test: {len(y_test_all)}")

# ============================================================
# 3. Simple Models
# ============================================================

print("\n" + "=" * 60)
print("Q3: Simple Models (Binary Classification)")
print("=" * 60)

perc_pre = Perceptron(random_state=SEED)
perc_pre.fit(X_train_avg_pre_bin, y_train_bin)
acc_perc_pre = accuracy_score(y_test_bin, perc_pre.predict(X_test_avg_pre_bin))
print(f"Perceptron (Pretrained W2V) Test Accuracy: {acc_perc_pre:.4f}")

perc_cus = Perceptron(random_state=SEED)
perc_cus.fit(X_train_avg_cus_bin, y_train_bin)
acc_perc_cus = accuracy_score(y_test_bin, perc_cus.predict(X_test_avg_cus_bin))
print(f"Perceptron (Custom W2V)     Test Accuracy: {acc_perc_cus:.4f}")

svm_pre = LinearSVC(random_state=SEED, max_iter=5000)
svm_pre.fit(X_train_avg_pre_bin, y_train_bin)
acc_svm_pre = accuracy_score(y_test_bin, svm_pre.predict(X_test_avg_pre_bin))
print(f"SVM (Pretrained W2V)        Test Accuracy: {acc_svm_pre:.4f}")

svm_cus = LinearSVC(random_state=SEED, max_iter=5000)
svm_cus.fit(X_train_avg_cus_bin, y_train_bin)
acc_svm_cus = accuracy_score(y_test_bin, svm_cus.predict(X_test_avg_cus_bin))
print(f"SVM (Custom W2V)            Test Accuracy: {acc_svm_cus:.4f}")

del perc_pre, perc_cus, svm_pre, svm_cus
gc.collect()


# ============================================================
# 4. Feedforward Neural Networks
# ============================================================

class FeedForwardMLP(nn.Module):
    """Feedforward MLP with two hidden layers (50 and 10 nodes)."""
    def __init__(self, input_dim, num_classes):
        super(FeedForwardMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_mlp(X_train, y_train, X_test, y_test, input_dim, num_classes,
              epochs=30, lr=0.001, batch_size=256):
    """Train an MLP and return the test accuracy."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label_min = y_train.min()
    y_tr = y_train - label_min
    y_te = y_test - label_min

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.LongTensor(y_tr)
    X_te_t = torch.FloatTensor(X_test)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))

    model = FeedForwardMLP(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_te_t), batch_size):
            batch = X_te_t[i:i+batch_size].to(device)
            preds = model(batch).argmax(dim=1).cpu()
            all_preds.append(preds)
    all_preds = torch.cat(all_preds).numpy()
    acc = accuracy_score(y_te, all_preds)

    del model, train_dataset, train_loader, X_tr_t, y_tr_t, X_te_t
    gc.collect()
    return acc


print("\n" + "=" * 60)
print("Q4(a): MLP with Average Word2Vec Features")
print("=" * 60)

acc_mlp_avg_pre_bin = train_mlp(
    X_train_avg_pre_bin, y_train_bin, X_test_avg_pre_bin, y_test_bin,
    input_dim=300, num_classes=2
)
print(f"MLP Avg (Pretrained) Binary  Accuracy: {acc_mlp_avg_pre_bin:.4f}")

acc_mlp_avg_pre_ter = train_mlp(
    X_train_avg_pre, y_train_all, X_test_avg_pre, y_test_all,
    input_dim=300, num_classes=3
)
print(f"MLP Avg (Pretrained) Ternary Accuracy: {acc_mlp_avg_pre_ter:.4f}")

acc_mlp_avg_cus_bin = train_mlp(
    X_train_avg_cus_bin, y_train_bin, X_test_avg_cus_bin, y_test_bin,
    input_dim=300, num_classes=2
)
print(f"MLP Avg (Custom)     Binary  Accuracy: {acc_mlp_avg_cus_bin:.4f}")

acc_mlp_avg_cus_ter = train_mlp(
    X_train_avg_cus, y_train_all, X_test_avg_cus, y_test_all,
    input_dim=300, num_classes=3
)
print(f"MLP Avg (Custom)     Ternary Accuracy: {acc_mlp_avg_cus_ter:.4f}")

# Free average features
del X_train_avg_pre, X_test_avg_pre, X_train_avg_cus, X_test_avg_cus
del X_train_avg_pre_bin, X_test_avg_pre_bin, X_train_avg_cus_bin, X_test_avg_cus_bin
gc.collect()

# 4(b) Concatenated features — compute and train one model at a time
print("\n" + "=" * 60)
print("Q4(b): MLP with Concatenated Word2Vec Features")
print("=" * 60)

print("Computing concatenated features (pretrained)...")
X_train_cat_pre = np.array([get_concat_w2v(t, pretrained_w2v) for t in train_df['tokens']])
X_test_cat_pre = np.array([get_concat_w2v(t, pretrained_w2v) for t in test_df['tokens']])
X_train_cat_pre_bin = X_train_cat_pre[train_binary_mask]
X_test_cat_pre_bin = X_test_cat_pre[test_binary_mask]

acc_mlp_cat_pre_bin = train_mlp(
    X_train_cat_pre_bin, y_train_bin, X_test_cat_pre_bin, y_test_bin,
    input_dim=3000, num_classes=2
)
print(f"MLP Concat (Pretrained) Binary  Accuracy: {acc_mlp_cat_pre_bin:.4f}")

acc_mlp_cat_pre_ter = train_mlp(
    X_train_cat_pre, y_train_all, X_test_cat_pre, y_test_all,
    input_dim=3000, num_classes=3
)
print(f"MLP Concat (Pretrained) Ternary Accuracy: {acc_mlp_cat_pre_ter:.4f}")

del X_train_cat_pre, X_test_cat_pre, X_train_cat_pre_bin, X_test_cat_pre_bin
gc.collect()

print("Computing concatenated features (custom)...")
X_train_cat_cus = np.array([get_concat_w2v(t, custom_w2v) for t in train_df['tokens']])
X_test_cat_cus = np.array([get_concat_w2v(t, custom_w2v) for t in test_df['tokens']])
X_train_cat_cus_bin = X_train_cat_cus[train_binary_mask]
X_test_cat_cus_bin = X_test_cat_cus[test_binary_mask]

acc_mlp_cat_cus_bin = train_mlp(
    X_train_cat_cus_bin, y_train_bin, X_test_cat_cus_bin, y_test_bin,
    input_dim=3000, num_classes=2
)
print(f"MLP Concat (Custom)     Binary  Accuracy: {acc_mlp_cat_cus_bin:.4f}")

acc_mlp_cat_cus_ter = train_mlp(
    X_train_cat_cus, y_train_all, X_test_cat_cus, y_test_all,
    input_dim=3000, num_classes=3
)
print(f"MLP Concat (Custom)     Ternary Accuracy: {acc_mlp_cat_cus_ter:.4f}")

del X_train_cat_cus, X_test_cat_cus, X_train_cat_cus_bin, X_test_cat_cus_bin
gc.collect()


# ============================================================
# 5. Convolutional Neural Networks (memory-efficient)
# ============================================================

class W2VSequenceDataset(Dataset):
    """PyTorch Dataset that builds Word2Vec sequences on-the-fly."""
    def __init__(self, token_lists, labels, w2v_model, max_len=50, dim=300):
        self.token_lists = token_lists
        self.labels = labels
        self.w2v_model = w2v_model
        self.max_len = max_len
        self.dim = dim

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx]
        vectors = [self.w2v_model[t] for t in tokens if t in self.w2v_model]
        if len(vectors) > self.max_len:
            vectors = vectors[:self.max_len]
        result = np.zeros((self.max_len, self.dim), dtype=np.float32)
        for i, v in enumerate(vectors):
            result[i] = v
        return torch.from_numpy(result), torch.tensor(self.labels[idx], dtype=torch.long)


class SentimentCNN(nn.Module):
    """Two-layer CNN for sentiment analysis."""
    def __init__(self, num_classes, embed_dim=300, max_len=50):
        super(SentimentCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=50, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=50, out_channels=10, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(10 * (max_len // 4), num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, embed_dim, max_len)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_cnn(train_tokens, y_train, test_tokens, y_test,
              w2v_model, num_classes, epochs=20, lr=0.001, batch_size=256):
    """Train a CNN using on-the-fly feature computation (memory efficient)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label_min = y_train.min()
    y_tr = (y_train - label_min).astype(np.int64)
    y_te = (y_test - label_min).astype(np.int64)

    train_dataset = W2VSequenceDataset(train_tokens, y_tr, w2v_model)
    test_dataset = W2VSequenceDataset(test_tokens, y_te, w2v_model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SentimentCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(batch_y)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)

    del model, train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    return acc


# Prepare token lists
train_tokens_all = train_df['tokens'].tolist()
test_tokens_all = test_df['tokens'].tolist()
train_tokens_bin = [train_tokens_all[i] for i in range(len(train_tokens_all)) if train_binary_mask[i]]
test_tokens_bin = [test_tokens_all[i] for i in range(len(test_tokens_all)) if test_binary_mask[i]]

print("\n" + "=" * 60)
print("Q5: CNN")
print("=" * 60)

acc_cnn_pre_bin = train_cnn(
    train_tokens_bin, y_train_bin, test_tokens_bin, y_test_bin,
    pretrained_w2v, num_classes=2
)
print(f"CNN (Pretrained) Binary  Accuracy: {acc_cnn_pre_bin:.4f}")

acc_cnn_pre_ter = train_cnn(
    train_tokens_all, y_train_all, test_tokens_all, y_test_all,
    pretrained_w2v, num_classes=3
)
print(f"CNN (Pretrained) Ternary Accuracy: {acc_cnn_pre_ter:.4f}")

acc_cnn_cus_bin = train_cnn(
    train_tokens_bin, y_train_bin, test_tokens_bin, y_test_bin,
    custom_w2v, num_classes=2
)
print(f"CNN (Custom)     Binary  Accuracy: {acc_cnn_cus_bin:.4f}")

acc_cnn_cus_ter = train_cnn(
    train_tokens_all, y_train_all, test_tokens_all, y_test_all,
    custom_w2v, num_classes=3
)
print(f"CNN (Custom)     Ternary Accuracy: {acc_cnn_cus_ter:.4f}")

# ============================================================
# Summary of All 16 Accuracy Values
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY OF ALL 16 ACCURACY VALUES")
print("=" * 70)

print("\n--- Q3: Simple Models (Binary) ---")
print(f"  1.  Perceptron (Pretrained W2V):     {acc_perc_pre:.4f}")
print(f"  2.  Perceptron (Custom W2V):         {acc_perc_cus:.4f}")
print(f"  3.  SVM (Pretrained W2V):            {acc_svm_pre:.4f}")
print(f"  4.  SVM (Custom W2V):                {acc_svm_cus:.4f}")

print("\n--- Q4(a): MLP with Average W2V ---")
print(f"  5.  MLP Avg (Pretrained) Binary:     {acc_mlp_avg_pre_bin:.4f}")
print(f"  6.  MLP Avg (Pretrained) Ternary:    {acc_mlp_avg_pre_ter:.4f}")
print(f"  7.  MLP Avg (Custom) Binary:         {acc_mlp_avg_cus_bin:.4f}")
print(f"  8.  MLP Avg (Custom) Ternary:        {acc_mlp_avg_cus_ter:.4f}")

print("\n--- Q4(b): MLP with Concatenated W2V ---")
print(f"  9.  MLP Concat (Pretrained) Binary:  {acc_mlp_cat_pre_bin:.4f}")
print(f"  10. MLP Concat (Pretrained) Ternary: {acc_mlp_cat_pre_ter:.4f}")
print(f"  11. MLP Concat (Custom) Binary:      {acc_mlp_cat_cus_bin:.4f}")
print(f"  12. MLP Concat (Custom) Ternary:     {acc_mlp_cat_cus_ter:.4f}")

print("\n--- Q5: CNN ---")
print(f"  13. CNN (Pretrained) Binary:         {acc_cnn_pre_bin:.4f}")
print(f"  14. CNN (Pretrained) Ternary:        {acc_cnn_pre_ter:.4f}")
print(f"  15. CNN (Custom) Binary:             {acc_cnn_cus_bin:.4f}")
print(f"  16. CNN (Custom) Ternary:            {acc_cnn_cus_ter:.4f}")

print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print("- CNN achieves the best binary accuracy, leveraging sequential word patterns.")
print("- Custom W2V outperforms pretrained for simple models & MLP (domain-specific advantage).")
print("- Average W2V features consistently beat concatenated features for MLP.")
print("- Ternary classification is significantly harder than binary across all models,")
print("  as the neutral class overlaps with both positive and negative sentiments.")
print("- Compared to HW1's bag-of-bigrams approach, Word2Vec-based models provide")
print("  dense semantic features that generalize better and extend to multi-class tasks.")
print("=" * 70)
