from dash import Dash, html, dcc
import dash

app = Dash(__name__, use_pages=True)

app.layout = html.Div([

    html.H1("Smart Hire", style={'textAlign': 'center'}),

    html.Div([
        dcc.Link("Home", href="/"),
        dcc.Link("Analysis", href="/analysis", style={'marginLeft': '20px'}),
        dcc.Link("Prediction", href="/prediction", style={'marginLeft': '20px'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dash.page_container
])
server = app.server
if __name__ == "__main__":
    app.run(debug=True)