"""
Supply Shed Visualizer - Refactored Main Application
A modular Dash application for visualizing supply shed data with interactive maps and charts.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_deck
import os
import io

# Import our modular components
from config import PROJECT_ID, DATASET_ID, TABLE_ID
from auth import register_auth_callbacks, create_login_page
from layouts import create_main_layout
from callbacks import register_all_callbacks
from data import initialize_bigquery_client

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Set app title
app.title = "Supply Shed Visualizer"

# Initialize BigQuery client
print("🚀 Initializing Supply Shed Visualizer...")
client = initialize_bigquery_client()

# Main app layout with authentication
app.layout = html.Div([
    dcc.Store(id='auth-state'),
    dcc.Store(id='main-content'),
    dcc.Interval(
        id='session-cleanup-interval',
        interval=5*60*1000,  # 5 minutes
        n_intervals=0
    ),
    html.Div(id='main-content')
])

# Register all callbacks
print("📝 Registering callbacks...")
register_auth_callbacks(app)
register_all_callbacks(app)

# Health check endpoint for Cloud Run
@app.server.route('/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return {'status': 'healthy', 'service': 'supply-shed-visualizer'}

# API endpoints for data streaming
@app.server.route('/api/data/facilities')
def get_facilities_data():
    """Stream facility data in chunks to avoid response size limits"""
    try:
        from data import get_data
        
        # Get the loaded data
        df, _ = get_data()
        
        # Return data in chunks
        chunk_size = 100  # Process 100 facilities at a time
        total_facilities = len(df)
        
        def generate():
            for i in range(0, total_facilities, chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                yield chunk.to_json(orient='records')
        
        return app.server.response_class(
            generate(),
            mimetype='application/json',
            headers={'Content-Type': 'application/json'}
        )
        
    except Exception as e:
        print(f"Error streaming facility data: {e}")
        return {'error': str(e)}, 500

@app.server.route('/api/data/plots')
def get_plots_data():
    """Stream plot data in chunks to avoid response size limits"""
    try:
        from data import get_data
        
        # Get the loaded data
        _, plot_df = get_data()
        
        # Return the plot data as JSON
        plots_data = plot_df.to_dict('records')
        
        # Check size and warn if large
        data_size_mb = len(str(plots_data)) / 1024 / 1024
        if data_size_mb > 10:
            print(f"⚠️  Large plot dataset: {data_size_mb:.1f} MB")
        
        return plots_data
        
    except Exception as e:
        print(f"Error streaming plot data: {e}")
        return {'error': str(e)}, 500

if __name__ == '__main__':
    # Get port from environment variable (for Cloud Run) or use default
    port = int(os.getenv('PORT', 8050))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Starting server on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📊 Project: {PROJECT_ID}")
    print(f"📊 Dataset: {DATASET_ID}")
    print(f"📊 Table: {TABLE_ID}")
    
    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
