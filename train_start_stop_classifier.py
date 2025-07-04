import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load labeled data
df = pd.read_csv("start_stop_labeled.csv")

X = df['text']
y = df['label']

# Only keep samples that are not "other" if you want a binary classifier,
# or keep all for a 3-class classifier.
# X = X[y != "other"]
# y = y[y != "other"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
print("Train accuracy:", clf.score(X_train_vec, y_train))
print("Test accuracy:", clf.score(X_test_vec, y_test))

# Save
joblib.dump(vectorizer, "startstop_vectorizer.joblib")
joblib.dump(clf, "startstop_classifier.joblib")
