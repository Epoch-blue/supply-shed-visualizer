"""
Supply Shed Visualizer - Main Application
A Dash application for visualizing supply shed data with interactive maps and charts.
"""
import os
import time
import json
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_deck
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from google.cloud import secretmanager

# Import our refactored modules
from src.config import EPOCH_COLORS, BIGQUERY_CONFIG, MAP_CONFIG
from src.auth import create_login_page, handle_login, get_session
from src.data import initialize_bigquery_client, fetch_supply_shed_data, fetch_facility_detail_data
from src.maps import create_deck_map, create_detail_map
from src.export import export_facility_data_as_geojson, export_plot_data_as_geoparquet, export_detail_map_data_as_geojson
from src.utils import create_facility_metadata_table, create_facility_label, get_highlighted_color, create_default_detail_map

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Supply Shed Visualizer"

# Global variables
bigquery_client = None
supply_shed_df = pd.DataFrame()
plot_df = pd.DataFrame()
supply_shed_detail_df = pd.DataFrame()

def get_mapbox_api_key():
    """Get Mapbox API key from Google Secret Manager or environment"""
    try:
        # Try to get from Secret Manager first
        client = secretmanager.SecretManagerServiceClient()
        project_id = BIGQUERY_CONFIG['PROJECT_ID']
        secret_name = f"projects/{project_id}/secrets/mapbox-api-key/versions/latest"
        
        response = client.access_secret_version(request={"name": secret_name})
        api_key = response.payload.data.decode("UTF-8")
        print("✅ Retrieved Mapbox API key from Secret Manager")
        return api_key
    except Exception as e:
        print(f"⚠️ Could not retrieve from Secret Manager: {e}")
        # Fallback to environment variable
        api_key = os.getenv("MAPBOX_API_KEY")
        if api_key:
            print("✅ Using Mapbox API key from environment variable")
            return api_key
        else:
            print("❌ No Mapbox API key found")
            return None

def create_main_layout():
    """Create the main application layout"""
    return html.Div([
        # Store components for data persistence
        dcc.Store(id='auth-state', data={'authenticated': False}),
        dcc.Store(id='session-id', data=None),
        dcc.Store(id='user-data', data=None),
        dcc.Store(id='supply-shed-data', data=None),
        dcc.Store(id='plot-data', data=None),
        dcc.Store(id='detail-map-data', data=None),
        dcc.Store(id='selected-points', data=[]),
        dcc.Store(id='highlighted-facility', data=None),
        dcc.Store(id='current-view-state', data=MAP_CONFIG['DEFAULT_VIEW_STATE']),
        dcc.Store(id='layer-toggle', data='facilities'),
        dcc.Store(id='map-style', data='dark'),
        dcc.Store(id='view-mode', data='3d'),
        dcc.Store(id='variable', data='total_tco2ehayear'),
        dcc.Store(id='detail-variable', data='total_tco2ehayear'),
        
        # Download components
        dcc.Download(id="main-map-download"),
        dcc.Download(id="detail-map-download"),
        
        # Main layout
        dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Supply Shed Visualizer", className="text-center mb-4", 
                           style={"color": EPOCH_COLORS['primary'], "fontWeight": "700"}),
                    html.P("Interactive visualization of supply shed data", className="text-center mb-4", 
                          style={"color": EPOCH_COLORS['text_secondary']})
                ], width=12)
            ]),
            
            # Metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Facilities", className="card-title"),
                            html.H2(id="total-facilities", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Plots", className="card-title"),
                            html.H2(id="total-plots", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Commodity Area (ha)", className="card-title"),
                            html.H2(id="total-commodity-area", className="text-warning")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Supply Shed Area (ha)", className="card-title"),
                            html.H2(id="total-supply-shed-area", className="text-info")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main content
            dbc.Row([
                # Left column - Main map
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H5("Supply Shed Overview", className="mb-0"),
                                html.Button(
                                    "{}",
                                    id="main-map-export-btn",
                                    className="btn btn-outline-primary btn-sm",
                                    style={"float": "right", "marginLeft": "10px", "fontFamily": "monospace", "lineHeight": "1"},
                                    title="Export current map data as GeoJSON"
                                )
                            ])
                        ]),
                        dbc.CardBody([
                            # Map controls
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Variable:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='variable-dropdown',
                                        options=[
                                            {'label': 'Total Emissions (tCO2e/ha/yr)', 'value': 'total_tco2ehayear'},
                                            {'label': 'Noncompliance Area (%)', 'value': 'noncompliance_area_perc'},
                                            {'label': 'Diversity Score', 'value': 'diversity_score'},
                                            {'label': 'Water Stress Index', 'value': 'water_stress_index'}
                                        ],
                                        value='total_tco2ehayear',
                                        clearable=False
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Layer:", className="fw-bold"),
                                    dbc.RadioItems(
                                        id='layer-toggle',
                                        options=[
                                            {'label': 'Facilities', 'value': 'facilities'},
                                            {'label': 'Plots', 'value': 'plots'}
                                        ],
                                        value='facilities',
                                        inline=True
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            # Main map
                            dcc.Loading(
                                id="main-map-loading",
                                children=[dash_deck.DeckGL(
                                    id='deck-map',
                                    data={},
                                    mapboxKey=get_mapbox_api_key(),
                                    style={'width': '100%', 'height': '500px'}
                                )],
                                type="default"
                            )
                        ])
                    ])
                ], width=8),
                
                # Right column - Charts and metadata
                dbc.Col([
                    # Facility metadata
                    dcc.Loading(
                        id="facility-metadata-loading",
                        children=[html.Div(id='facility-metadata')],
                        type="default"
                    ),
                    
                    # Charts
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("Facility Analysis", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='facility-chart')
                        ])
                    ], className="mt-3")
                ], width=4)
            ], className="mb-4"),
            
            # Detail map section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H5("Supply Shed Detail", className="mb-0"),
                                html.Button(
                                    "{}",
                                    id="detail-map-export-btn",
                                    className="btn btn-outline-primary btn-sm",
                                    style={"float": "right", "marginLeft": "10px", "fontFamily": "monospace", "lineHeight": "1"},
                                    title="Export detail map data as GeoJSON"
                                )
                            ])
                        ]),
                        dbc.CardBody([
                            # Detail map controls
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Variable:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='detail-variable-dropdown',
                                        options=[
                                            {'label': 'Total Emissions (tCO2e/ha/yr)', 'value': 'total_tco2ehayear'},
                                            {'label': 'Noncompliance Area (%)', 'value': 'noncompliance_area_perc'},
                                            {'label': 'Diversity Score', 'value': 'diversity_score'},
                                            {'label': 'Water Stress Index', 'value': 'water_stress_index'}
                                        ],
                                        value='total_tco2ehayear',
                                        clearable=False
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            # Detail map
                            dcc.Loading(
                                id="detail-map-loading",
                                children=[dash_deck.DeckGL(
                                    id='detail-map',
                                    data={},
                                    mapboxKey=get_mapbox_api_key(),
                                    style={'width': '100%', 'height': '500px'}
                                )],
                                type="default"
                            )
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)
    ])

# Main app layout
app.layout = html.Div([
    dcc.Store(id='auth-state', data={'authenticated': False}),
    dcc.Store(id='session-id', data=None),
    dcc.Store(id='user-data', data=None),
    html.Div(id='main-content', children=create_login_page())
])

if __name__ == '__main__':
    # Get port from environment (for Cloud Run)
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Starting Supply Shed Visualizer on port {port}")
    print(f"🔧 Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
