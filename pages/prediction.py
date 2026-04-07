import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import re
import base64
import io
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
df = df[df['Category'].astype(str).str.strip() != ""]
df['Category'] = df['Category'].astype(str)
df['cleaned'] = df['cleaned'].fillna("")

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['Category']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ---------------- DOMAIN SKILLS ----------------
role_skills = {
    "information technology": ["python", "java", "sql", "machine learning", "html", "css", "javascript"],
    "human resources": ["recruitment", "employee", "training", "hr", "communication"],
    "consultant": ["analysis", "strategy", "client", "business", "presentation"],
    "designer": ["photoshop", "figma", "ui", "ux", "design"],
    "sales": ["sales", "marketing", "negotiation", "client", "targets"]
}

def extract_score(text, domain):
    text = text.lower()
    skills = role_skills[domain]
    match = sum(1 for s in skills if s in text)
    return (match / len(skills)) * 100

def extract_skills(text):
    text = text.lower()
    found = []
    for skills in role_skills.values():
        for s in skills:
            if s in text and s not in found:
                found.append(s)
    return found

# ---------------- PDF ----------------
def extract_text_from_pdf(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    pdf = PdfReader(io.BytesIO(decoded))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# ---------------- UI ----------------
layout = html.Div(className="container", children=[

    html.Div(className="card", children=[

        html.H2("Resume Selection"),

        dcc.Dropdown(
            id='domain',
            options=[
                {"label": "IT", "value": "information technology"},
                {"label": "HR", "value": "human resources"},
                {"label": "Consultant", "value": "consultant"},
                {"label": "Designer", "value": "designer"},
                {"label": "Sales", "value": "sales"},
            ],
            placeholder="Select Domain"
        ),

        html.Br(),

        dcc.Upload(
            id='upload_pdf',
            children=html.Div(['📄 Upload PDFs']),
            multiple=True,
            style={'border': '2px dashed #38bdf8', 'padding': '20px'}
        ),

        html.Div(id='file_list'),

        html.Br(),
        html.Button("Select Best Candidate", id='btn'),

        html.H3(id='selected_candidate'),

        html.Div(id='result_table'),
        dcc.Graph(id='chart'),
    ])
])

# ---------------- FILE LIST ----------------
@dash.callback(
    Output('file_list', 'children'),
    Input('upload_pdf', 'filename')
)
def show_files(names):
    if not names:
        return "No files uploaded"
    return html.Ul([html.Li(f"📄 {n}") for n in names])

# ---------------- MAIN ----------------
@dash.callback(
    Output('result_table', 'children'),
    Output('chart', 'figure'),
    Output('selected_candidate', 'children'),
    Input('btn', 'n_clicks'),
    State('domain', 'value'),
    State('upload_pdf', 'contents'),
    State('upload_pdf', 'filename')
)
def compare(n, domain, pdf_contents, filenames):

    if not n or not domain:
        return "", {}, "⚠️ Select domain and upload resumes"

    resumes = []
    names = []

    if pdf_contents:
        for i, content in enumerate(pdf_contents[:5]):
            try:
                resumes.append(extract_text_from_pdf(content))
                names.append(filenames[i])
            except:
                continue

    results = []

    for i, res in enumerate(resumes):
        cleaned = clean_text(res)
        vec = vectorizer.transform([cleaned])
        prob = float(model.predict_proba(vec).max() * 100)

        domain_score = extract_score(res, domain)
        final_score = (0.6 * domain_score) + (0.4 * prob)

        skills = extract_skills(res)

        results.append({
            "Name": names[i],
            "Score": final_score,
            "Skills": ", ".join(skills)
        })

    results = sorted(results, key=lambda x: x['Score'], reverse=True)

    best = results[0]["Name"] if results else ""
    selected_text = f"🏆 Selected Candidate: {best}"

    table = html.Table([
        html.Tr([html.Th("Candidate"), html.Th("Score"), html.Th("Skills")])
    ] + [
        html.Tr([
            html.Td(r["Name"]),
            html.Td(f"{r['Score']:.2f}"),
            html.Td(r["Skills"])
        ]) for r in results
    ])

    fig = go.Figure(data=[
        go.Bar(x=[r["Name"] for r in results],
               y=[r["Score"] for r in results])
    ])

    fig.update_layout(
        title="Candidate Ranking",
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font=dict(color='white')
    )

    return table, fig, selected_text
