"""
Layout module for the Supply Shed Visualizer application.
Contains all UI layout components and page structures.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_deck
# Removed config import as it doesn't exist

def create_main_layout():
    """Create the main application layout"""
    return html.Div([
        # Header
        dbc.NavbarSimple(
            brand="Supply Shed Visualizer",
            brand_href="#",
            color="dark",
            dark=True,
            className="mb-4"
        ),
        
        # Main content container
        dbc.Container([
            # Metrics row
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
                            html.H4("Total Area (ha)", className="card-title"),
                            html.H2(id="total-area", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Avg tCO2e/ha/year", className="card-title"),
                            html.H2(id="avg-emissions", className="text-warning")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Avg Diversity Score", className="card-title"),
                            html.H2(id="avg-diversity", className="text-info")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main content row
            dbc.Row([
                # Left column - Main map and controls
                dbc.Col([
                    # Map controls
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Variable:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='y-axis-dropdown',
                                        options=[
                                            {'label': 'Total tCO2e/ha/year', 'value': 'total_tco2ehayear'},
                                            {'label': 'LUC tCO2e/ha/year', 'value': 'luc_tco2ehayear'},
                                            {'label': 'Non-LUC tCO2e/ha/year', 'value': 'nonluc_tco2ehayear'},
                                            {'label': 'Diversity Score', 'value': 'diversity_score'},
                                            {'label': 'Water Stress Index', 'value': 'water_stress_index'},
                                            {'label': 'Precipitation/PET Ratio', 'value': 'precipitation_pet_ratio'},
                                            {'label': 'Soil Moisture Percentile', 'value': 'soil_moisture_percentile'},
                                            {'label': 'Evapotranspiration Anomaly', 'value': 'evapotranspiration_anomaly'},
                                            {'label': 'Area (ha) Supply Shed', 'value': 'area_ha_supply_shed'},
                                            {'label': 'Estate/Smallholder Ratio', 'value': 'estate_smallholder_ratio'},
                                            {'label': 'Non-compliance Area %', 'value': 'noncompliance_area_perc'}
                                        ],
                                        value='total_tco2ehayear',
                                        clearable=False
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Layer:", className="fw-bold"),
                                    daq.ToggleSwitch(
                                        id='layer-toggle',
                                        label='Plots',
                                        labelPosition='top',
                                        value=False
                                    )
                                ], width=6)
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Main map container
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                # Export button
                                html.Div([
                                    html.Button(
                                        "{}",
                                        id="export-main-map-button",
                                        className="btn btn-outline-primary btn-sm",
                                        style={
                                            "position": "absolute",
                                            "top": "10px",
                                            "right": "10px",
                                            "zIndex": "1000",
                                            "fontSize": "16px",
                                            "lineHeight": "1",
                                            "width": "30px",
                                            "height": "30px",
                                            "borderRadius": "50%",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center"
                                        },
                                        title="Export current map data as GeoJSON file"
                                    ),
                                    # Export loading spinner
                                    html.Div(
                                        id="export-main-map-spinner",
                                        style={"display": "none", "position": "absolute", "top": "10px", "right": "50px", "zIndex": "1000"},
                                        children=[
                                            html.Div(
                                                style={
                                                    "width": "20px",
                                                    "height": "20px",
                                                    "border": "2px solid #f3f3f3",
                                                    "borderTop": "2px solid #3498db",
                                                    "borderRadius": "50%",
                                                    "animation": "spin 1s linear infinite"
                                                }
                                            )
                                        ]
                                    )
                                ], style={"position": "relative"}),
                                
                                # Map
                                dash_deck.DeckGL(
                                    id='deck-map',
                                    data={'layers': []},
                                    mapboxKey=get_mapbox_api_key(),
                                    style={'height': '500px'}
                                ),
                                
                                # Loading spinner
                                dbc.Spinner(
                                    html.Div(id="main-map-spinner"),
                                    id="main-map-loading",
                                    spinner_style={"width": "3rem", "height": "3rem"},
                                    color="primary"
                                )
                            ], style={"position": "relative"})
                        ])
                    ])
                ], width=8),
                
                # Right column - Chart and metadata
                dbc.Col([
                    # Chart
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='main-chart')
                        ])
                    ], className="mb-3"),
                    
                    # Facility metadata
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Facility Details", className="card-title"),
                            html.Div(id="facility-metadata")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Detail map row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                # Export button
                                html.Div([
                                    html.Button(
                                        "{}",
                                        id="export-detail-map-button",
                                        className="btn btn-outline-primary btn-sm",
                                        style={
                                            "position": "absolute",
                                            "top": "10px",
                                            "right": "10px",
                                            "zIndex": "1000",
                                            "fontSize": "16px",
                                            "lineHeight": "1",
                                            "width": "30px",
                                            "height": "30px",
                                            "borderRadius": "50%",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center"
                                        },
                                        title="Export detail map data as GeoJSON file"
                                    ),
                                    # Export loading spinner
                                    html.Div(
                                        id="export-detail-map-spinner",
                                        style={"display": "none", "position": "absolute", "top": "10px", "right": "50px", "zIndex": "1000"},
                                        children=[
                                            html.Div(
                                                style={
                                                    "width": "20px",
                                                    "height": "20px",
                                                    "border": "2px solid #f3f3f3",
                                                    "borderTop": "2px solid #3498db",
                                                    "borderRadius": "50%",
                                                    "animation": "spin 1s linear infinite"
                                                }
                                            )
                                        ]
                                    )
                                ], style={"position": "relative"}),
                                
                                # Detail map
                                dash_deck.DeckGL(
                                    id='detail-map',
                                    data={'layers': []},
                                    mapboxKey=get_mapbox_api_key(),
                                    style={'height': '400px'}
                                ),
                                
                                # Loading spinner
                                dbc.Spinner(
                                    html.Div(id="detail-loading-container"),
                                    id="detail-map-loading",
                                    spinner_style={"width": "3rem", "height": "3rem"},
                                    color="primary"
                                )
                            ], style={"position": "relative"})
                        ])
                    ])
                ], width=8),
                
                # Detail map controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Color Variable:", className="fw-bold"),
                            dcc.Dropdown(
                                id='detail-color-dropdown',
                                options=[
                                    {'label': 'Total tCO2e/ha/year', 'value': 'total_tco2ehayear'},
                                    {'label': 'LUC tCO2e/ha/year', 'value': 'luc_tco2ehayear'},
                                    {'label': 'Non-LUC tCO2e/ha/year', 'value': 'nonluc_tco2ehayear'},
                                    {'label': 'Diversity Score', 'value': 'diversity_score'},
                                    {'label': 'Water Stress Index', 'value': 'water_stress_index'},
                                    {'label': 'Precipitation/PET Ratio', 'value': 'precipitation_pet_ratio'},
                                    {'label': 'Soil Moisture Percentile', 'value': 'soil_moisture_percentile'},
                                    {'label': 'Evapotranspiration Anomaly', 'value': 'evapotranspiration_anomaly'},
                                    {'label': 'Area (ha)', 'value': 'area_ha'},
                                    {'label': 'Non-compliance Area %', 'value': 'noncompliance_area_perc'}
                                ],
                                value='total_tco2ehayear',
                                clearable=False
                            )
                        ])
                    ])
                ], width=4)
            ])
        ], fluid=True),
        
        # Hidden stores for data persistence
        dcc.Store(id='highlighted-facility'),
        dcc.Store(id='detail-map-data'),
        dcc.Store(id='auth-state'),
        
        # Session cleanup interval
        dcc.Interval(
            id='session-cleanup-interval',
            interval=5*60*1000,  # 5 minutes
            n_intervals=0
        ),
        
        # Custom CSS for spinners and tooltips
        html.Div([
            html.Script("""
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    [title]:hover::after {
                        content: attr(title);
                        position: absolute;
                        bottom: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 5px 10px;
                        border-radius: 4px;
                        font-size: 12px;
                        white-space: nowrap;
                        z-index: 1000;
                        pointer-events: none;
                    }
                    
                    [title]:hover::before {
                        content: '';
                        position: absolute;
                        bottom: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        border: 5px solid transparent;
                        border-top-color: rgba(0, 0, 0, 0.8);
                        z-index: 1000;
                        pointer-events: none;
                    }
                `;
                document.head.appendChild(style);
            """)
        ], style={"display": "none"})
    ])

# Redundant function removed - use the one from app.py
