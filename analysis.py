import dash
from dash import html, dcc
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

dash.register_page(__name__)

# ---------------- LOAD DATA ----------------
df = pd.read_excel("data/resume.xlsx")

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['cleaned'] = df['Resume_str'].apply(clean_text)
df = df.dropna(subset=['Category'])
df['Category'] = df['Category'].astype(str).str.lower().str.strip()

# ---------------- MAP TO YOUR DOMAINS ----------------
mapping = {
    "information technology": "IT",
    "it": "IT",
    "human resources": "HR",
    "hr": "HR",
    "consultant": "Consultant",
    "designer": "Designer",
    "sales": "Sales"
}

df['Mapped'] = df['Category'].map(mapping)

# keep only required domains
df = df[df['Mapped'].notna()]

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['Mapped']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- ROC ----------------
classes = ["IT", "HR", "Consultant", "Designer", "Sales"]

# keep only classes present in test set
classes = [c for c in classes if c in list(y_test)]

y_test_bin = label_binarize(y_test, classes=classes)
y_score = model.predict_proba(X_test)

fig = go.Figure()

for i in range(len(classes)):
    try:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f"{classes[i]} (AUC={roc_auc:.2f})"
        ))
    except:
        continue

# diagonal
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    line=dict(dash='dash'),
    name='Random'
))

fig.update_layout(
    title="ROC Curve (HR / IT / Consultant / Designer / Sales)",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    plot_bgcolor='#1e293b',
    paper_bgcolor='#1e293b',
    font=dict(color='white')
)

# ---------------- UI ----------------
layout = html.Div(className="container", children=[

    html.Div(className="card", children=[
        html.H2("Model Evaluation"),
        html.P("ROC Curve for selected hiring domains."),
        dcc.Graph(figure=fig)
    ])

])