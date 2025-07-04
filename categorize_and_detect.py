import joblib

cat_vectorizer = joblib.load("category_vectorizer.joblib")
cat_clf = joblib.load("category_classifier.joblib")
ss_vectorizer = joblib.load("startstop_vectorizer.joblib")
ss_clf = joblib.load("startstop_classifier.joblib")

def categorize_statement(statement):
    X = cat_vectorizer.transform([statement])
    return cat_clf.predict(X)[0]

def detect_start_stop(statement):
    X = ss_vectorizer.transform([statement])
    return ss_clf.predict(X)[0]

# Example usage:
print(categorize_statement("Yesterday I fixed a bug in the deployment script."))
print(categorize_statement("Today I will work on the new API endpoints."))
print(categorize_statement("I'm blocked by a missing permission."))

print(detect_start_stop("I'm done"))
print(detect_start_stop("Let me begin"))
print(detect_start_stop("That's all"))
