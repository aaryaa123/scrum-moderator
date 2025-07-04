import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("category_labeled.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df['label'])

y_pred = clf.predict(X_test)
print(classification_report(test_df['label'], y_pred))

joblib.dump(vectorizer, "category_vectorizer.joblib")
joblib.dump(clf, "category_classifier.joblib")
