import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("data/resume.xlsx")

df['Category'] = df['Category'].astype(str).str.lower().str.strip()

df['Category'] = df['Category'].replace({
    'buisness devop': 'business development',
    'public relationship': 'public relations',
    'data media': 'data science',
    'it': 'information technology',
    'hr': 'human resources'
})

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['cleaned'] = df['Resume_str'].apply(clean_text)

df = df[df['cleaned'] != ""]
df = df.dropna(subset=['Category'])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])

y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))