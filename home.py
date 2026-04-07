import dash
from dash import html

dash.register_page(__name__, path="/")

layout = html.Div(className="container", children=[

    # ---------------- INTRO ----------------
    html.Div(className="card", children=[
        html.H2("Smart Hire"),
        html.P("An intelligent system to analyze resumes and select the best candidate efficiently."),
        html.P("Upload resumes, choose a domain, and let the system rank candidates based on skills and suitability.")
    ]),

    # ---------------- GUIDELINES ----------------
    html.Div(className="card", children=[
        html.H2("Selection Guidelines"),

        html.P("The system evaluates candidates based on domain-specific skills and machine learning-based suitability scoring."),

        html.Ul([
            html.Li("Above 80% → Strong Candidate (High probability of selection)"),
            html.Li("60% – 80% → Good Candidate (Competitive profile)"),
            html.Li("50% – 60% → Moderate Candidate (Needs improvement)"),
            html.Li("Below 50% → Low Chances (Less suitable for the role)")
        ]),

        html.P("Candidates are compared relative to each other within the selected domain."),

        html.P("Note: The system provides decision support and does not guarantee selection, but helps recruiters make informed choices.")
    ])

])