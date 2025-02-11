from collections import Counter
from itertools import chain

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("quotes.csv")
print(df.head())
print(df.info())
print(df.isnull().count())
print(df.isna().count())
df.dropna(inplace=True)

label_counts = Counter(chain(*df["category"]))

# Keep only labels that appear at least 50 times (adjust as needed)
min_label_freq = 50
common_labels = {
    label for label, count in label_counts.items() if count >= min_label_freq
}

# Remove rare labels from dataset
labels = df["category"].apply(
    lambda labels: [label for label in labels if label in common_labels]
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["quote"])
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

print(X)
print(y)
