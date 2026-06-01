import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "Issue": [
        "System crash issue",
        "Payment failed",
        "Password reset needed",
        "Website loading slowly",
        "Account hacked",
        "Unable to access dashboard"
    ],

    "Category": [
        "Technical",
        "Billing",
        "Support",
        "Technical",
        "Security",
        "Support"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Issue"])
y = df["Category"]

# Train model
classifier = MultinomialNB()
classifier.fit(X, y)

# New tickets for prediction
test_tickets = [
    "Cannot access my account",
    "Payment not completed",
    "Website is very slow"
]

# Transform test data
X_test = vectorizer.transform(test_tickets)

# Predict categories
predicted_categories = classifier.predict(X_test)

# Display results
for issue, category in zip(test_tickets, predicted_categories):
    print(f"Issue: {issue}")
    print(f"Predicted Category: {category}")
    print("-" * 30)