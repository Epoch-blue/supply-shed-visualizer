import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import secretmanager
import os
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
import dash_deck
import dash_daq as daq
import pydeck as pdk
import json
import io
import geopandas as gpd
from shapely.geometry import shape
import secrets

# Custom JSON encoder to handle NaN values
class SafeJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        # Recursively clean NaN values from nested objects
        obj = self._clean_nan_values(obj)
        return super().encode(obj)
    
    def _clean_nan_values(self, obj):
        if isinstance(obj, dict):
            return {key: self._clean_nan_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan_values(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj

def get_mapbox_api_key():
    """Get Mapbox API key from Google Secret Manager"""
    try:
        # Initialize the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()
        
        # Project ID (you may need to adjust this)
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "epoch-geospatial-dev")
        
        # Secret name
        secret_name = "mapbox-api-key"
        
        # Build the resource name of the secret version
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        
        # Access the secret version
        response = client.access_secret_version(request={"name": name})
        
        # Return the secret value
        return response.payload.data.decode("UTF-8")
        
    except Exception as e:
        print(f"Error fetching Mapbox API key from Secret Manager: {e}")
        print("Falling back to environment variable...")
        
        # Fallback to environment variable
        load_dotenv()
        return os.getenv("MAPBOX_API_KEY")

# Get Mapbox API key from Secret Manager (with fallback to env var)
mapbox_api_token = get_mapbox_api_key()

# Simple authentication configuration
VALID_CREDENTIALS = {
    'william@epoch.blue': 'ssi123'
}

# In-memory session storage (in production, use Redis or database)
user_sessions = {}

def validate_credentials(username, password):
    """Validate user credentials"""
    return VALID_CREDENTIALS.get(username) == password

# Initialize the Dash app with Epoch-like theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
])
app.title = "Supply Shed Visualizer | Epoch"

# Login page layout
def create_login_page():
    """Create the login page layout"""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        # Epoch logo/branding
                        html.Div([
                            html.H1("Supply Shed Visualizer", className="text-center mb-4", 
                                   style={"color": EPOCH_COLORS['primary'], "fontWeight": "700"}),
                            html.P("Powered by Epoch", className="text-center mb-5", 
                                  style={"color": EPOCH_COLORS['text_secondary'], "fontSize": "18px"})
                        ], className="mb-5"),
                        
                        # Login card
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H4("Welcome", className="text-center mb-4", 
                                           style={"color": EPOCH_COLORS['text_primary']}),
                                    html.P("Please sign in to access the Supply Shed Visualizer", 
                                          className="text-center mb-4", 
                                          style={"color": EPOCH_COLORS['text_secondary']}),
                                    
                                    # Login form
                                    dbc.Form([
                                        # Username field
                                        dbc.Label("Email", html_for="username-input", className="mb-2"),
                                        dbc.Input(
                                            id="username-input",
                                            type="email",
                                            placeholder="Enter your email",
                                            className="mb-3",
                                            style={"height": "48px"}
                                        ),
                                        
                                        # Password field
                                        dbc.Label("Password", html_for="password-input", className="mb-2"),
                                        dbc.Input(
                                            id="password-input",
                                            type="password",
                                            placeholder="Enter your password",
                                            className="mb-4",
                                            style={"height": "48px"}
                                        ),
                                        
                                        # Login button
                                        dbc.Button(
                                            "Sign In",
                                            id="login-btn",
                                            color="primary",
                                            size="lg",
                                            className="w-100",
                                            style={"height": "48px", "fontWeight": "500"}
                                        )
                                    ]),
                                    
                                    # Loading spinner (hidden by default)
                                    html.Div([
                                        dbc.Spinner(html.Div(), size="sm", color="primary")
                                    ], id="login-spinner", style={"display": "none"}, className="text-center mt-3"),
                                    
                                    # Error message (hidden by default)
                                    html.Div(id="login-error", style={"display": "none"}, className="mt-3")
                                    
                                ], style={"padding": "2rem"})
                            ])
                        ], style={"maxWidth": "400px", "margin": "0 auto", "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"})
                        
                    ], style={"minHeight": "100vh", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
                ], width=12)
            ])
        ], fluid=True)
    ], style={"backgroundColor": EPOCH_COLORS['background'], "minHeight": "100vh"})

# Epoch theme colors
EPOCH_COLORS = {
    'primary': '#376fd0',      # customBlue[700]
    'secondary': '#4782da',    # customBlue[500]
    'accent': '#D9B430',       # epoch yellow (accent color)
    'background': '#F7F9FC',   # light gray background
    'paper': '#FFFFFF',        # white paper
    'text_primary': '#233044', # dark blue-gray
    'text_secondary': '#6c757d', # gray
    'epoch_yellow': '#D9B430', # epoch yellow
    'sidebar': '#233044',      # dark sidebar
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8'
}

# BigQuery configuration
PROJECT_ID = os.getenv('BIGQUERY_PROJECT_ID', 'epoch-geospatial-dev')
DATASET_ID = os.getenv('BIGQUERY_DATASET_ID', '1mUTPmnLDbWCneHliVw34sAe1ck1')
TABLE_ID = os.getenv('BIGQUERY_TABLE_ID', 'stat_supply_shed')

# Initialize BigQuery client with proper credentials handling
def initialize_bigquery_client():
    """Initialize BigQuery client with proper credentials for production"""
    try:
        # Check if we're running in Cloud Run (production)
        if os.getenv('K_SERVICE'):  # Cloud Run sets this environment variable
            print("🚀 Running in Cloud Run - using default credentials")
            # In Cloud Run, use the service account attached to the service
            client = bigquery.Client(project=PROJECT_ID)
        else:
            # Local development - try service account file first, then default credentials
            service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if service_account_path and os.path.exists(service_account_path):
                print(f"🔧 Using service account file: {service_account_path}")
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
            else:
                print("🔧 Using default credentials")
                client = bigquery.Client(project=PROJECT_ID)
        
        # Test the connection
        test_query = f"SELECT 1 as test FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` LIMIT 1"
        client.query(test_query).result()
        print(f"✅ BigQuery client initialized successfully for project: {PROJECT_ID}")
        return client
        
    except Exception as e:
        print(f"❌ BigQuery client initialization failed: {e}")
        print("   Make sure the service account has proper permissions:")
        print("   - BigQuery Data Viewer")
        print("   - BigQuery Job User")
        print("   - Secret Manager Secret Accessor")
        return None

# Initialize BigQuery client
client = initialize_bigquery_client()

def fetch_data():
    """Fetch data from BigQuery"""
    if not client:
        return create_sample_data()
    
    query = f"""
    SELECT 
        * EXCEPT(area_ha_supply_shed),
        ST_AREA(geo) / 10000 as area_ha_supply_shed,
        SAFE_DIVIDE(area_ha_estate, area_ha_smallholder) as estate_smallholder_ratio,
        ST_X(ST_GEOGFROMTEXT(facility_geo)) as longitude,
        ST_Y(ST_GEOGFROMTEXT(facility_geo)) as latitude
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE facility_geo IS NOT NULL
    """
    
    try:
        df = client.query(query).to_dataframe()
        # Clean the data immediately after fetching
        df = clean_dataframe(df)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return create_sample_data()

def fetch_facility_detail_data(facility_id, company_name, df):
    """Fetch detailed plot and supply shed data for a specific facility"""
    try:
        # Get the collection_id from the existing dataframe using both facility_id and company_name
        if facility_id and company_name:
            # Try to match both facility_id and company_name
            facility_row = df[(df['facility_id'] == facility_id) & (df['company_name'] == company_name)]
        elif facility_id:
            # Fallback to just facility_id
            facility_row = df[df['facility_id'] == facility_id]
        elif company_name:
            # Fallback to just company_name
            facility_row = df[df['company_name'] == company_name]
        else:
            print("No facility_id or company_name provided")
            return pd.DataFrame(), pd.DataFrame()
        
        if facility_row.empty:
            print(f"No facility found with ID: {facility_id} and company: {company_name}")
            return pd.DataFrame(), pd.DataFrame()
        
        collection_id = facility_row.iloc[0]['collection_id']
        print(f"Found collection_id: {collection_id} for facility: {facility_id}, company: {company_name}")
        
        # Query plot data
        plot_query = f"""
        SELECT 
            noncompliance_area_ha,
            noncompliance_area_perc,
            luc_tco2eyear,
            nonluc_tco2eyear,
            luc_tco2ehayear,
            nonluc_tco2ehayear,
            diversity_score,
            water_stress_index,
            luc_tco2eyear + nonluc_tco2eyear as total_tco2eyear,
            luc_tco2ehayear + nonluc_tco2ehayear as total_tco2ehayear,
            area_ha,
            ST_ASGEOJSON(geo) as geometry
        FROM `{PROJECT_ID}.{DATASET_ID}.{collection_id}_plot`
        WHERE geo IS NOT NULL 
        """
        
        # Query supply shed data
        supply_shed_query = f"""
        SELECT 
            uuid,
            ST_ASGEOJSON(geo) as geometry
        FROM `{PROJECT_ID}.{DATASET_ID}.{collection_id}_supply_shed`
        WHERE geo IS NOT NULL
        """
        
        # Execute queries
        plot_job = client.query(plot_query)
        supply_shed_job = client.query(supply_shed_query)
        
        plot_results = plot_job.result()
        supply_shed_results = supply_shed_job.result()
        
        # Convert to DataFrames
        plot_df = pd.DataFrame([dict(row) for row in plot_results])
        supply_shed_df = pd.DataFrame([dict(row) for row in supply_shed_results])
        
        # Clean the dataframes to ensure JSON serializability
        print(f"Plot columns: {list(plot_df.columns)}")
        print(f"Supply shed columns: {list(supply_shed_df.columns)}")
        
        # More aggressive cleaning for detailed data
        # Replace NaN, inf, and -inf with None
        plot_df = plot_df.replace([np.nan, np.inf, -np.inf], None)
        supply_shed_df = supply_shed_df.replace([np.nan, np.inf, -np.inf], None)
        
        # Also use where to catch any remaining null values
        plot_df = plot_df.where(pd.notnull(plot_df), None)
        supply_shed_df = supply_shed_df.where(pd.notnull(supply_shed_df), None)
        
        # Convert any remaining numpy types
        for col in plot_df.columns:
            if plot_df[col].dtype == 'object':
                continue
            elif plot_df[col].dtype in ['int64', 'int32']:
                plot_df[col] = plot_df[col].astype('int64')
            elif plot_df[col].dtype in ['float64', 'float32']:
                plot_df[col] = plot_df[col].astype('float64')
        
        for col in supply_shed_df.columns:
            if supply_shed_df[col].dtype == 'object':
                continue
            elif supply_shed_df[col].dtype in ['int64', 'int32']:
                supply_shed_df[col] = supply_shed_df[col].astype('int64')
            elif supply_shed_df[col].dtype in ['float64', 'float32']:
                supply_shed_df[col] = supply_shed_df[col].astype('float64')
        
        print(f"Fetched {len(plot_df)} plot records and {len(supply_shed_df)} supply shed records")
        return plot_df, supply_shed_df
        
    except Exception as e:
        print(f"Error fetching facility detail data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def clean_dataframe(df):
    """Clean dataframe to ensure JSON serializability"""
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace NaN values in numeric columns
    numeric_columns = ['noncompliance_area_rate', 'total_tco2ehayear', 'diversity_score', 'water_stress_index', 'area_ha']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Ensure coordinates are valid numbers (only if they exist)
    if 'longitude' in df_clean.columns:
        df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce').fillna(0)
    if 'latitude' in df_clean.columns:
        df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce').fillna(0)
    
    # Remove rows with invalid coordinates (only if coordinate columns exist)
    coord_columns = [col for col in ['longitude', 'latitude'] if col in df_clean.columns]
    if coord_columns:
        df_clean = df_clean.dropna(subset=coord_columns)
    
    # Replace all NaN values with None (which becomes null in JSON)
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    # Convert numpy types to native Python types for JSON serialization
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            continue
        elif df_clean[col].dtype in ['int64', 'int32']:
            df_clean[col] = df_clean[col].astype('int64')
        elif df_clean[col].dtype in ['float64', 'float32']:
            df_clean[col] = df_clean[col].astype('float64')
    
    return df_clean

def create_sample_data():
    """Create sample data for development/testing"""
    np.random.seed(42)
    n_points = 200
    
    # Generate more realistic coordinates (North America focus)
    lats = np.random.uniform(25.0, 50.0, n_points)
    lons = np.random.uniform(-125.0, -65.0, n_points)
    
    data = {
        'longitude': lons,
        'latitude': lats,
        'noncompliance_area_rate': np.random.uniform(0, 1, n_points),
        'total_tco2ehayear': np.random.uniform(0, 1000, n_points),
        'diversity_score': np.random.uniform(0, 10, n_points),
        'water_stress_index': np.random.uniform(0, 5, n_points),
        'id': range(n_points)
    }
    
    df = pd.DataFrame(data)
    return clean_dataframe(df)

def fetch_plot_hexagon_data():
    """Fetch raw plot data from stat_plot table"""
    try:
        # Fetch raw plot data with lat/lon coordinates
        query = f"""
        SELECT 
            ST_X(ST_CENTROID(geo)) as longitude,
            ST_Y(ST_CENTROID(geo)) as latitude,
            noncompliance_area_ha,
            noncompliance_area_perc,
            luc_tco2eyear + nonluc_tco2eyear as total_tco2eyear,            
            luc_tco2ehayear + nonluc_tco2ehayear as total_tco2ehayear,
            diversity_score,
            water_stress_index,
            area_ha
        FROM `{PROJECT_ID}.{DATASET_ID}.stat_plot`
        WHERE ST_X(ST_CENTROID(geo)) IS NOT NULL 
          AND ST_Y(ST_CENTROID(geo)) IS NOT NULL
        --LIMIT 100000  -- Limit to prevent frontend crashes
        """
        # Use streaming for large results
        query_job = client.query(query)
        hexagon_df = query_job.to_dataframe(create_bqstorage_client=True)
    
        # Clean the data
        hexagon_df = hexagon_df.dropna(subset=['longitude', 'latitude'])
        
        # Handle NaN values in numeric columns
        numeric_columns = hexagon_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            hexagon_df[col] = hexagon_df[col].fillna(0)
        
        print(f"Loaded {len(hexagon_df)} raw plot records")
        if len(hexagon_df) > 0:
            print(f"Sample plot data: {hexagon_df.head(3).to_dict()}")
            print(f"Plot data columns: {list(hexagon_df.columns)}")
            print(f"Coordinate ranges - Lat: {hexagon_df['latitude'].min():.2f} to {hexagon_df['latitude'].max():.2f}")
            print(f"Coordinate ranges - Lon: {hexagon_df['longitude'].min():.2f} to {hexagon_df['longitude'].max():.2f}")
            print(f"First 3 coordinates: {hexagon_df[['longitude', 'latitude']].head(3).to_dict()}")
        return hexagon_df
    except Exception as e:
        print(f"Error fetching hexagon data: {e}")
        return pd.DataFrame()

# Load data after all functions are defined but before layout creation
print("Loading facility data...")
df = fetch_data()
print(f"Loaded {len(df)} facilities")
# Plot data will be loaded lazily when needed
plot_df = None

# Mapbox API key

def create_deck_map(selected_points=None, 
                    map_style='mapbox://styles/mapbox/dark-v11', 
                    variable='total_tco2ehayear', 
                    view_mode='3d', 
                    current_view_state=None,
                    highlighted_facility=None
                    ):
    """Create deck.gl map configuration"""
    # Always show all data (no filtering by selected_points)
    map_data = df.copy()
    
    # Clean the data - replace NaN values with 0 or appropriate defaults
    numeric_columns = ['noncompliance_area_rate', 'total_tco2ehayear', 'diversity_score', 'water_stress_index', 'area_ha_commodity']
    for col in numeric_columns:
        if col in map_data.columns:
            map_data[col] = map_data[col].fillna(0)
    
    # Ensure longitude and latitude are valid numbers
    map_data['longitude'] = pd.to_numeric(map_data['longitude'], errors='coerce').fillna(0)
    map_data['latitude'] = pd.to_numeric(map_data['latitude'], errors='coerce').fillna(0)
    
    # Remove any rows with invalid coordinates
    map_data = map_data.dropna(subset=['longitude', 'latitude'])
    
    # Create color ramp function for selected variable using quantile stretching (green -> yellow -> red)
    def get_color_ramp(value):
        """Convert value to color ramp using quantile stretching with appropriate direction based on field type"""
        # Use quantile stretching (2nd and 98th percentiles) to handle outliers better
        q2 = map_data[variable].quantile(0.02)
        q98 = map_data[variable].quantile(0.98)
        
        if q98 == q2:
            normalized = 0.5
        else:
            # Clamp value to quantile range and normalize
            clamped_value = max(q2, min(q98, value))
            normalized = (clamped_value - q2) / (q98 - q2)
        
        # For diversity_score, invert the color ramp (high = good = green)
        if 'diversity' in variable.lower():
            normalized = 1 - normalized  # Invert for diversity
        
        # Green to Yellow to Red color ramp
        if normalized <= 0.5:
            # Green to Yellow
            ratio = normalized * 2
            r = int(255 * ratio)
            g = 255
            b = 0
        else:
            # Yellow to Red
            ratio = (normalized - 0.5) * 2
            r = 255
            g = int(255 * (1 - ratio))
            b = 0
        
        return [r, g, b, 200]  # Add alpha for transparency
    
    # Add color column to the data
    map_data['color'] = map_data[variable].apply(get_color_ramp)
    
    # Highlight clicked facility if specified
    if highlighted_facility:
        print(f"Highlighting facility: {highlighted_facility}")  # Debug print
        # Create a highlighted color (bright yellow) for the highlighted facility
        def get_highlighted_color(row):
            # Create the same facility_company label as in the chart
            facility_id = row.get('facility_id', 'Unknown')
            company_name = row.get('company_name', 'Unknown')
            
            # Clean the values
            if isinstance(facility_id, str):
                facility_id = facility_id.strip()
            if isinstance(company_name, str):
                company_name = company_name.strip()
            
            # Check if values are meaningful
            def is_meaningful(value):
                if pd.isna(value):
                    return False
                if isinstance(value, str) and value.strip().upper() in ['NA', 'N/A', 'NULL', '']:
                    return False
                return True
            
            facility_meaningful = is_meaningful(facility_id)
            company_meaningful = is_meaningful(company_name)
            
            if facility_meaningful and company_meaningful:
                facility_company = f"{facility_id} - {company_name}"
            elif facility_meaningful:
                facility_company = str(facility_id)
            elif company_meaningful:
                facility_company = str(company_name)
            else:
                facility_company = "Unknown"
            
            if facility_company == highlighted_facility:
                return [255, 255, 0, 255]  # Bright yellow for highlighting
            else:
                return get_color_ramp(row[variable])
        
        map_data['color'] = map_data.apply(get_highlighted_color, axis=1)
        # Add highlight flag
        map_data['is_highlighted'] = map_data.apply(lambda row: row['color'] == [255, 255, 0, 255], axis=1)
    else:
        map_data['is_highlighted'] = False
    
    # Convert to dict and ensure all values are JSON serializable
    map_data_dict = map_data.to_dict('records')
    for record in map_data_dict:
        for key, value in record.items():
            # Skip color arrays - they're already properly formatted
            if key == 'color':
                continue
            elif pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                record[key] = 0
            elif isinstance(value, (np.integer, np.floating)):
                # Round numeric values to 3 decimal places
                if isinstance(value, np.floating):
                    record[key] = round(float(value), 3)
                else:
                    record[key] = value.item()
    
    # Create layers - one for normal points and one for highlighted points
    layers = []
    
    # Normal points layer
    normal_data = [point for point in map_data_dict if not point.get('is_highlighted', False)]
    if normal_data:
        normal_layer = pdk.Layer(
            "ScatterplotLayer",
            data=normal_data,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="area_ha_commodity",
            pickable=True,
            auto_highlight=True,
            radius_scale=0.1,
            radius_min_pixels=3,
            radius_max_pixels=50,
            get_customdata=["facility_id"],
            extruded=True
        )
        layers.append(normal_layer)
    
    # Highlighted points layer (on top) - use HexagonLayer for more visibility
    highlighted_data = [point for point in map_data_dict if point.get('is_highlighted', False)]
    if highlighted_data:
        
        # Also add a regular scatterplot layer for the highlighted points
        highlighted_scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=highlighted_data,
            get_position=["longitude", "latitude"],
            get_fill_color=[255, 255, 0, 255],  # Bright yellow
            get_radius="area_ha_commodity",
            pickable=True,
            auto_highlight=True,
            radius_scale=0.25,  # Much larger for emphasis
            radius_min_pixels=10,
            radius_max_pixels=100,
            get_customdata=["facility_id"],
            extruded=True,
            elevation_scale=50
        )
        layers.append(highlighted_scatter_layer)
    
    # Create view state - use custom view state if provided (for zooming to hovered facility)
    if current_view_state:
        view_state = pdk.ViewState(
            latitude=current_view_state['latitude'],
            longitude=current_view_state['longitude'],
            zoom=current_view_state['zoom'],
            pitch=current_view_state['pitch'],
            bearing=current_view_state['bearing'],
            min_zoom=1,
            max_zoom=20
        )
    elif len(map_data) > 0:
        # Calculate bounds of the data
        min_lat = map_data['latitude'].min()
        max_lat = map_data['latitude'].max()
        min_lon = map_data['longitude'].min()
        max_lon = map_data['longitude'].max()
        
        # Calculate center point
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Calculate zoom level based on data extent
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        max_range = max(lat_range, lon_range)
        
        # Adjust zoom based on data spread (smaller spread = higher zoom) - increased by 1 level
        if max_range > 10:
            zoom = 4
        elif max_range > 5:
            zoom = 5
        elif max_range > 2:
            zoom = 6
        elif max_range > 1:
            zoom = 7
        else:
            zoom = 8
        
        # Determine pitch based on view mode
        if view_mode == '3d':
            pitch = 45  # Force 3D view
        else:
            pitch = 0  # Force flat 2D view
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom-0.5,
            pitch=pitch,
            bearing=0,
            min_zoom=1,
            max_zoom=20
        )
    else:
        # Fallback values if no data
        view_state = pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=2,
            pitch=45 if view_mode == '3d' else 0,
            bearing=0,
            min_zoom=1,
            max_zoom=20
        )
    
    # Map style configuration
    map_style_config = {
        'dark': "mapbox://styles/mapbox/dark-v11", 
        'light': "mapbox://styles/mapbox/light-v11",
        'road': "mapbox://styles/mapbox/standard",
        'satellite': "mapbox://styles/mapbox/satellite-v9"
    }
    
    # Create deck configuration with proper map setup
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style_config.get(map_style, "mapbox://styles/mapbox/dark-v11"),
        api_keys={"mapbox": mapbox_api_token},
        height=500,
        width="100%",
        tooltip=True
    )
    
    return deck.to_json()

# Epoch theme colors
EPOCH_COLORS = {
    'primary': '#376fd0',      # customBlue[700]
    'secondary': '#4782da',    # customBlue[500]
    'accent': '#D9B430',       # epoch yellow (accent color)
    'background': '#F7F9FC',   # light gray background
    'paper': '#FFFFFF',        # white paper
    'text_primary': '#233044', # dark blue-gray
    'text_secondary': '#6c757d', # gray
    'epoch_yellow': '#D9B430', # epoch yellow
    'sidebar': '#233044',      # dark sidebar
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8'
}

# Define the layout with Epoch styling and authentication
app.layout = html.Div([
    # Store for authentication state
    dcc.Store(id='auth-state', data={'authenticated': False, 'user': None}),
    dcc.Store(id='session-id', data=None),
    
    # Main content area - start with login page
    html.Div(id='main-content', children=create_login_page())
], style={'backgroundColor': EPOCH_COLORS['background']})

# Main application layout (shown when authenticated)
def create_main_layout():
    """Create the main application layout"""
    return html.Div([
        # Header
        html.Div([
        html.Div([
            # Logo and title row
            html.Div([
                html.Div([
                    html.Img(
                        src="assets/epoch-logo-text-light.png",
                        style={'height': '60px', 'width': 'auto'},
                        alt="Epoch Logo"
                    )
                ], className="col-auto"),
                html.Div([
                    html.H1("Supply Shed Visualizer", className="epoch-title", style={'margin': '0', 'fontSize': '2.5rem', 'color': 'white'}),
                    html.P("Enviromental Metrics of All Palm Mills Supply Sheds in Indonesia", className="epoch-subtitle", style={'margin': '0', 'fontSize': '1.1rem', 'color': 'white'})
                ], className="col text-center"),
                html.Div([
                    html.Img(
                        src="assets/wwf.png",
                        style={'height': '80px', 'width': 'auto'},
                        alt="WWF Logo"
                    )
                ], className="col-auto")
            ], className="row align-items-center justify-content-between")
        ], className="container")
    ], className="epoch-header"),
    
    # Main content
    dbc.Container([
        # Top metrics row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(id="metric-total-facilities", className="epoch-metric-value"),
                    html.P("Total Facilities (#)", className="epoch-metric-label")
                ], className="epoch-metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H3(id="metric-total-plots", className="epoch-metric-value"),
                    html.P("Total Plots (#)", className="epoch-metric-label")
                ], className="epoch-metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H3(id="metric-supply-shed-area", className="epoch-metric-value"),
                    html.P("Total Supply Shed Area (ha)", className="epoch-metric-label")
                ], className="epoch-metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H3(id="metric-commodity-area", className="epoch-metric-value"),
                    html.P("Total Commodity Area (ha)", className="epoch-metric-label")
                ], className="epoch-metric-card")
            ], width=3)
        ], className="mb-4"),
        
        # Map row - split into main map (3/4) and detail map (1/4)
        dbc.Row([
            # Main map (3/4 width)
            dbc.Col([
                    html.Div([
                    html.Div([
                        html.H5("Supply Shed Overview", className="epoch-title mb-0"),
                        # Export button and spinner for main map
                        html.Div([
                            html.Button(
                                "{}",
                                id="main-map-export-btn",
                                className="btn btn-outline-primary btn-sm",
                                style={
                                    "width": "35px",
                                    "height": "35px",
                                    "borderRadius": "50%",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "backgroundColor": "rgba(255,255,255,0.9)",
                                    "border": "1px solid #007bff",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                    "fontFamily": "monospace",
                                    "lineHeight": "1"
                                },
                                title="Export current map data as GeoJSON file"
                            ),
                            # Spinner for export
                            html.Div(
                                id="main-map-export-spinner",
                                style={
                                    "marginLeft": "10px",
                                    "width": "16px",
                                    "height": "16px",
                                    "border": "2px solid #f3f3f3",
                                    "borderTop": "2px solid #007bff",
                                    "borderRadius": "50%",
                                    "animation": "spin 1s linear infinite",
                                    "display": "none"
                                }
                            )
                        ], style={"display": "flex", "alignItems": "center"})
                    ], className="d-flex align-items-center justify-content-between epoch-card-header"),
                    html.Div([
                        dcc.Dropdown(
                            id='y-axis-dropdown',
                            options=[
                                {'label': 'Total Emissions (tCO2e/ha/yr)', 'value': 'total_tco2ehayear'},
                                {'label': 'Noncompliance Area (ha)', 'value': 'noncompliance_area_ha'},
                                {'label': 'Noncompliance Area (%)', 'value': 'noncompliance_area_perc'},
                                {'label': 'Diversity Score', 'value': 'diversity_score'},
                                {'label': 'Water Stress Index', 'value': 'water_stress_index'},
                                {'label': 'Transpiration Anomaly', 'value': 'transpiration_anomaly'},
                                {'label': 'Estate/Smallholder Ratio', 'value': 'estate_smallholder_ratio'},
                                {'label': 'Area (ha) - Estate', 'value': 'area_ha_estate'},
                                {'label': 'Area (ha) - Smallholder', 'value': 'area_ha_smallholder'},
                                {'label': 'Area (ha) - Total', 'value': 'area_ha_total'},
                                {'label': 'Area (ha) - Supply Shed', 'value': 'area_ha_supply_shed'},
                                {'label': 'Area (ha) - Commodity', 'value': 'area_ha_commodity'}
                            ],
                            value='total_tco2ehayear',
                            clearable=False,
                            className="epoch-dropdown",
                            style={'width': '300px', 'fontSize': '12px'}
                        )
                    ], style={'padding': '0.5rem 1.5rem 0 1.5rem'}),
                    html.Div([
                        # Map container
                        html.Div([
                            # Layer toggle and loading spinner in top-right
                            html.Div([
                                # Layer toggle slider and export button
                                html.Div([
                                    html.Label("Facilities", style={"fontSize": "12px", "color": "#666", "marginRight": "8px"}),
                                    daq.ToggleSwitch(
                                        id='main-map-layer-toggle',
                                        value=False,  # False = Facilities, True = Plots
                                        color="#007bff",
                                        size=40,
                                        style={"margin": "0 8px"}
                                    ),
                                    html.Label("Plots", style={"fontSize": "12px", "color": "#666", "marginLeft": "8px"})
                                ], style={
                                    "position": "absolute", 
                                    "top": "10px", 
                                    "right": "10px", 
                                    "zIndex": "1000",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "backgroundColor": "rgba(255,255,255,0.9)",
                                    "padding": "8px 12px",
                                    "borderRadius": "20px",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
                                })
                            ], style={"position": "relative"}),
                            # The actual map with loading wrapper
                            dcc.Loading(
                                id="main-map-loading",
                                type="default",
                                children=dash_deck.DeckGL(
                                    id='deck-map',
                                    data=create_deck_map(),
                                    style={'height': '500px', 'width': '100%'},
                                    mapboxKey=mapbox_api_token,
                                    enableEvents=['click']
                                ),
                                style={"position": "relative"}
                            ),
                            # JavaScript to optimize canvas performance and add spinner animation
                            html.Script("""
                                setTimeout(function() {
                                    const canvas = document.querySelector('#deck-map canvas');
                                    if (canvas) {
                                        canvas.willReadFrequently = true;
                                    }
                                }, 1000);
                                
                                // Add CSS animation for spinner and tooltips
                                const style = document.createElement('style');
                                style.textContent = `
                                    @keyframes spin {
                                        0% { transform: rotate(0deg); }
                                        100% { transform: rotate(360deg); }
                                    }
                                    
                                    /* Custom tooltip styles */
                                    [title] {
                                        position: relative;
                                    }
                                    
                                    [title]:hover::after {
                                        content: attr(title);
                                        position: absolute;
                                        bottom: 100%;
                                        left: 50%;
                                        transform: translateX(-50%);
                                        background-color: #333;
                                        color: white;
                                        padding: 8px 12px;
                                        border-radius: 4px;
                                        font-size: 12px;
                                        white-space: nowrap;
                                        z-index: 1000;
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                                        margin-bottom: 5px;
                                    }
                                    
                                    [title]:hover::before {
                                        content: '';
                                        position: absolute;
                                        bottom: 100%;
                                        left: 50%;
                                        transform: translateX(-50%);
                                        border: 5px solid transparent;
                                        border-top-color: #333;
                                        z-index: 1000;
                                        margin-bottom: -5px;
                                    }
                                `;
                                document.head.appendChild(style);
                            """)
                        ], className="map-container")
                    ], style={'padding': '1.5rem'})
                ], className="epoch-card")
            ], width=8),
            
            # Detail map (1/3 width)
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("Supply Shed Detail", className="epoch-title mb-0"),
                        # Export button and spinner for detail map
                        html.Div([
                            html.Button(
                                "{}",
                                id="detail-map-export-btn",
                                className="btn btn-outline-primary btn-sm",
                                style={
                                    "width": "35px",
                                    "height": "35px",
                                    "borderRadius": "50%",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "backgroundColor": "rgba(255,255,255,0.9)",
                                    "border": "1px solid #007bff",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                    "fontFamily": "monospace",
                                    "lineHeight": "1"
                                },
                                title="Export detail map data as GeoJSON file"
                            ),
                            # Spinner for export
                            html.Div(
                                id="detail-map-export-spinner",
                                style={
                                    "marginLeft": "10px",
                                    "width": "16px",
                                    "height": "16px",
                                    "border": "2px solid #f3f3f3",
                                    "borderTop": "2px solid #007bff",
                                    "borderRadius": "50%",
                                    "animation": "spin 1s linear infinite",
                                    "display": "none"
                                }
                            )
                        ], style={"display": "flex", "alignItems": "center"})
                    ], className="d-flex align-items-center justify-content-between epoch-card-header"),
                    html.Div([
                        dbc.Spinner(
                            html.Div(id="detail-loading-content"),
                            id="detail-loading-spinner",
                            color="primary",
                            size="sm",
                            spinner_style={"width": "1rem", "height": "1rem"}
                        )
                    ], style={"display": "none"}, id="detail-loading-container"),
                    html.Div([
                        dcc.Dropdown(
                            id='detail-color-dropdown',
                            options=[
                                {'label': 'Noncompliance Area (ha)', 'value': 'noncompliance_area_ha'},
                                {'label': 'Noncompliance Area (%)', 'value': 'noncompliance_area_perc'},
                                {'label': 'LUC Emissions (tCO2e/yr)', 'value': 'luc_tco2eyear'},
                                {'label': 'LUC Emissions (tCO2e/ha/yr)', 'value': 'luc_tco2ehayear'},
                                {'label': 'Non-LUC Emissions (tCO2e/yr)', 'value': 'nonluc_tco2eyear'},
                                {'label': 'Non-LUC Emissions (tCO2e/ha/yr)', 'value': 'nonluc_tco2ehayear'},
                                {'label': 'Total Emissions (tCO2e/yr)', 'value': 'total_tco2eyear'},
                                {'label': 'Total Emissions (tCO2e/ha/yr)', 'value': 'total_tco2ehayear'},
                                {'label': 'Diversity Score', 'value': 'diversity_score'},
                                {'label': 'Precipitation-PET Ratio', 'value': 'precipitation_pet_ratio'},
                                {'label': 'Soil Moisture Percentile', 'value': 'soil_moisture_percentile'},
                                {'label': 'Evapotranspiration Anomaly', 'value': 'evapotranspiration_anomaly'},
                                {'label': 'Water Stress Index', 'value': 'water_stress_index'}
                            ],
                            value='total_tco2ehayear',
                            className="epoch-dropdown",
                            style={'width': '300px', 'fontSize': '12px'}
                        )
                    ], style={'padding': '0.5rem 1.5rem 0 1.5rem'}),
                    html.Div([
                        html.Div([
                            dcc.Loading(
                                id="detail-map-loading",
                                type="default",
                                children=dash_deck.DeckGL(
                                    id='detail-map',
                                    data={'layers': []},
                                    style={'height': '500px', 'width': '100%'},
                                    mapboxKey=mapbox_api_token,
                                    tooltip={
                                        "html": "<b>Plot ID:</b> {plot_id}<br/>"
                                               "<b>Plot Area:</b> {area_ha} ha<br/>"
                                               "<b>Deforestation Area (ha):</b> {noncompliance_area_ha} ha<br/>"
                                               "<b>Deforestation Area (%):</b> {noncompliance_area_perc} ha<br/>"
                                               "<b>LUC Emissions (tCO2 / ha / yr):</b> {total_tco2ehayear} tCO2e/year<br/>"
                                               "<b>Diversity Score:</b> {diversity_score}<br/>"
                                               "<b>Water Stress Index:</b> {water_stress_index}",
                                        "style": {
                                            "backgroundColor": "steelblue",
                                            "color": "white",
                                            "padding": "10px",
                                            "borderRadius": "5px"
                                        }
                                    }
                                ),
                                style={"position": "relative"}
                            )
                        ], className="map-container")
                    ], style={'padding': '1.5rem'})
                ], className="epoch-card")
            ], width=4)
        ], className="mb-4"),
        
        # Bottom row: Chart (2/3) + Facility metadata (1/3)
        dbc.Row([
            # Chart (2/3 width)
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("All Facilities Chart", className="epoch-title mb-0")
                    ], className="d-flex align-items-center justify-content-between epoch-card-header"),
                    html.Div([
                        dcc.Graph(id='main-chart')
                    ], style={'padding': '1.5rem'})
                ], className="epoch-card")
            ], width=8),
            
            # Facility metadata (1/3 width)
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("Facility Metadata", className="epoch-title mb-2")
                    ], className="d-flex align-items-center justify-content-between epoch-card-header"),
                    html.Div([
                        dcc.Loading(
                            id="facility-metadata-loading",
                            type="default",
                            children=html.Div(id='facility-metadata', children=[
                                html.P("Click on a chart bar or map point to view facility details", 
                                       className="text-muted text-center", 
                                       style={'padding': '2rem', 'fontStyle': 'italic'})
                            ]),
                            style={"position": "relative"}
                        )
                    ], style={'padding': '1.5rem'})
                ], className="epoch-card")
            ], width=4)
        ], className="mb-4")
    ], fluid=True, style={'backgroundColor': EPOCH_COLORS['background'], 'minHeight': '100vh'}),
    
    # Store for selected points
    dcc.Store(id='selected-points', data=[]),
    
    # Store for hovered facility (like the dash_leaflet example)
    dcc.Store(id='hovered-facility', data=None),
    # Store for highlighted facility (clicked feature)
    dcc.Store(id='highlighted-facility', data=None),
    # Store for detail map data (plot and supply shed dataframes)
    dcc.Store(id='detail-map-data', data=None),
    # Download components for exports
    dcc.Download(id="main-map-download"),
    dcc.Download(id="detail-map-download")
])

# Authentication callbacks
@app.callback(
    Output('main-content', 'children'),
    [Input('auth-state', 'data')],
    prevent_initial_call=True
)
def update_main_content(auth_state):
    """Update main content based on authentication state"""
    if auth_state and auth_state.get('authenticated'):
        return create_main_layout()
    else:
        return create_login_page()

@app.callback(
    [Output('auth-state', 'data'),
     Output('session-id', 'data'),
     Output('login-error', 'children'),
     Output('login-error', 'style')],
    [Input('login-btn', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value')],
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password):
    """Handle login form submission"""
    if n_clicks and username and password:
        if validate_credentials(username, password):
            # Successful login
            session_id = secrets.token_urlsafe(32)
            user_sessions[session_id] = {
                'authenticated': True,
                'user': {
                    'email': username,
                    'name': username.split('@')[0].title(),
                    'picture': None
                }
            }
            return (
                {'authenticated': True, 'user': user_sessions[session_id]['user']}, 
                session_id,
                "",
                {"display": "none"}
            )
        else:
            # Failed login
            error_msg = dbc.Alert(
                "Invalid email or password. Please try again.",
                color="danger",
                dismissable=True
            )
            return (
                {'authenticated': False, 'user': None}, 
                None,
                error_msg,
                {"display": "block"}
            )
    
    return {'authenticated': False, 'user': None}, None, "", {"display": "none"}

# Combined callback to update highlighted facility, metadata, and detail map from both chart and map clicks
@app.callback(
    [Output('highlighted-facility', 'data'),
     Output('facility-metadata', 'children'),
     Output('detail-map', 'data', allow_duplicate=True),
     Output('detail-map-data', 'data'),
     Output('detail-loading-container', 'style')],
    [Input('main-chart', 'clickData'),
     Input('deck-map', 'clickInfo')],
    prevent_initial_call=True
)
def update_highlight_metadata_and_detail_from_clicks(chart_click_data, map_click_data):
    """Update highlighted facility, metadata, and detail map when chart or map is clicked"""
    ctx = dash.callback_context
    
    # Add a small delay to make loading visible
    import time
    time.sleep(0.3)  # 300ms delay to show loading spinner
    
    if not ctx.triggered:
        return None, None, {'layers': []}, None, {"display": "none"}
    
    trigger_id = ctx.triggered[0]['prop_id']
    facility_id = None
    facility_label = None
    
    # Show loading indicator immediately
    loading_style = {"display": "block"}
    
    # Handle chart click
    if 'main-chart' in trigger_id and chart_click_data and 'points' in chart_click_data:
        facility_label = chart_click_data['points'][0]['x']
        # Extract facility ID from the label (format: "FACILITY_ID - COMPANY_NAME")
        if ' - ' in facility_label:
            facility_id = facility_label.split(' - ')[0]
        else:
            facility_id = facility_label
    
    # Handle map click
    elif 'deck-map' in trigger_id and map_click_data:
        print(f"Map click data: {map_click_data}")
        # Try different possible structures for the click data
        if 'object' in map_click_data and map_click_data['object'] is not None:
            obj = map_click_data['object']
            facility_id = obj.get('facility_id', '')
            company_name = obj.get('company_name', '')
            if facility_id:
                print(f"Found facility_id from object: {facility_id}")
                # Create facility label for highlighting
                if company_name and company_name != 'Unknown':
                    facility_label = f"{facility_id} - {company_name}"
                else:
                    facility_label = facility_id
        elif 'layer' in map_click_data and 'object' in map_click_data['layer'] and map_click_data['layer']['object'] is not None:
            obj = map_click_data['layer']['object']
            facility_id = obj.get('facility_id', '')
            company_name = obj.get('company_name', '')
            if facility_id:
                print(f"Found facility_id from layer.object: {facility_id}")
                if company_name and company_name != 'Unknown':
                    facility_label = f"{facility_id} - {company_name}"
                else:
                    facility_label = facility_id
        elif isinstance(map_click_data, dict):
            # Try to find facility_id in any nested structure
            for key, value in map_click_data.items():
                if isinstance(value, dict) and 'facility_id' in value:
                    facility_id = value['facility_id']
                    company_name = value.get('company_name', '')
                    if facility_id:
                        print(f"Found facility_id in {key}: {facility_id}")
                        if company_name and company_name != 'Unknown':
                            facility_label = f"{facility_id} - {company_name}"
                        else:
                            facility_label = facility_id
                        break
    
    # If we found a facility, update metadata and load detail map
    if facility_id:
        # Create metadata table
        facility_row = df[df['facility_id'] == facility_id]
        if not facility_row.empty:
            facility_data = facility_row.iloc[0]
            metadata_table = create_facility_metadata_table(facility_data)
        else:
            metadata_table = html.P("Facility data not found", className="text-danger")
        
        # Load detail map data
        detail_map_data = fetch_facility_detail_data(facility_id, None, df)
        if detail_map_data[0].empty and detail_map_data[1].empty:
            detail_map_layers = {'layers': []}
            stored_data = None
        else:
            detail_map_layers = create_detail_map(detail_map_data[0], detail_map_data[1])
            # Get collection_id from facility data
            collection_id = facility_row.iloc[0]['collection_id'] if not facility_row.empty else None
            # Store the dataframes and collection_id for the color dropdown and export callbacks
            stored_data = {
                'plot_df': detail_map_data[0].to_dict('records'),
                'supply_shed_df': detail_map_data[1].to_dict('records'),
                'collection_id': collection_id
            }
        return facility_label, metadata_table, detail_map_layers, stored_data, {"display": "none"}
    
    return None, None, {'layers': []}, None, loading_style

# Callback to load default detail map data on app startup
@app.callback(
    [Output('detail-map', 'data'),
     Output('detail-map-data', 'data', allow_duplicate=True)],
    Input('detail-map', 'id'),
    prevent_initial_call=True
)
def load_default_detail_map(_):
    """Load default detail map data on app startup"""
    detail_map_data = create_default_detail_map()
    
    # Also populate the detail-map-data store for the dropdown callback
    if detail_map_data and len(detail_map_data) == 2:
        # Use the default collection ID
        default_collection_id = "0x706310441550789c3dc429e3098592f3702e07e8b36cae7c06e96daf1a85a65d"
        stored_data = {
            'plot_df': detail_map_data[0].to_dict('records'),
            'supply_shed_df': detail_map_data[1].to_dict('records'),
            'collection_id': default_collection_id
        }
        return create_detail_map(detail_map_data[0], detail_map_data[1]), stored_data
    else:
        return {'layers': []}, None


def get_default_metrics():
    """Get default metrics from database tables"""
    try:
        # Get total facilities count
        total_facilities = len(df)
        
        # Get total plots count from stat_plot table
        plot_count_query = f"""
        SELECT COUNT(*) as total_plots, SUM(ST_AREA(geo) / 10000) as total_commodity_area
        FROM `{PROJECT_ID}.{DATASET_ID}.stat_plot`
        """
        plot_count_result = client.query(plot_count_query).to_dataframe()
        
        total_plots = plot_count_result['total_plots'].iloc[0] if not plot_count_result.empty else 0
        total_commodity_area = plot_count_result['total_commodity_area'].iloc[0] if not plot_count_result.empty else 0
        
        
        # Handle NaN values
        if pd.isna(total_commodity_area):
            total_commodity_area = 0

        # Get total supply shed area from stat_supply_shed table
        supply_shed_area_query = f"""
        SELECT ST_AREA(ST_UNION_AGG(geo)) / 10000 as total_supply_shed_area
        FROM `{PROJECT_ID}.{DATASET_ID}.stat_supply_shed`
        """
        supply_shed_area_result = client.query(supply_shed_area_query).to_dataframe()
        total_supply_shed_area = supply_shed_area_result['total_supply_shed_area'].iloc[0] if not supply_shed_area_result.empty else 0
        
        # Handle NaN values
        if pd.isna(total_supply_shed_area):
            total_supply_shed_area = 0
        
        final_metrics = {
            'total_facilities': total_facilities,
            'total_plots': total_plots,
            'total_commodity_area': total_commodity_area,
            'total_supply_shed_area': total_supply_shed_area
        }
        return final_metrics
    except Exception as e:
        print(f"Error getting default metrics: {e}")
        return {
            'total_facilities': len(df),
            'total_plots': 0,
            'total_commodity_area': 0,
            'total_supply_shed_area': 0
        }


def get_color_ramp(df, variable):
    """Get color ramp for a given variable"""
    if variable not in df.columns:
        return [255, 0, 0, 255]  # Default red color
    
    values = df[variable].dropna()
    if len(values) == 0:
        return [255, 0, 0, 255]  # Default red color
    
    # Use 5th and 95th percentiles for color stretching
    min_val = values.quantile(0.05)
    max_val = values.quantile(0.95)
    
    if min_val == max_val:
        return [255, 0, 0, 255]  # Default red color
    
    def get_color(value):
        if pd.isna(value):
            return [128, 128, 128, 255]  # Gray for NaN
        
        # Normalize value between 0 and 1
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))  # Clamp between 0 and 1
        
        # Invert color ramp for diversity-related fields
        if 'diversity' in variable.lower():
            normalized = 1 - normalized
        
        # Create color ramp: Red (low) -> Orange -> Yellow -> Green (high)
        if normalized < 0.33:
            # Red to Orange
            r = 255
            g = int(255 * normalized / 0.33)
            b = 0
        elif normalized < 0.66:
            # Orange to Yellow
            r = int(255 * (0.66 - normalized) / 0.33)
            g = 255
            b = 0
        else:
            # Yellow to Green
            r = 0
            g = 255
            b = int(255 * (normalized - 0.66) / 0.34)
        
        return [r, g, b, 255]
    
    return get_color

def create_plot_hexagon_layer(plot_df, variable='total_tco2ehayear'):
    """Create a hexagon layer for pre-aggregated plot data"""
    if plot_df.empty:
        print("WARNING: No plot data available for hexagon layer")
        return {
            "@@type": "ScatterplotLayer",
            "data": [],
            "id": "plot-hexagon-layer"
        }
    
    print(f"HEXAGON: Creating plot layer with {len(plot_df)} points for variable: {variable}")
    print(f"HEXAGON: Variable column exists in data: {variable in plot_df.columns}")
    if variable in plot_df.columns:
        print(f"HEXAGON: Variable {variable} range: {plot_df[variable].min():.3f} to {plot_df[variable].max():.3f}")
    
    # Debug: Check if we have valid coordinates
    if len(plot_df) > 0:
        print(f"Data validation - Lat range: {plot_df['latitude'].min():.2f} to {plot_df['latitude'].max():.2f}")
        print(f"Data validation - Lon range: {plot_df['longitude'].min():.2f} to {plot_df['longitude'].max():.2f}")
        print(f"Data validation - First point: Lat={plot_df.iloc[0]['latitude']:.2f}, Lon={plot_df.iloc[0]['longitude']:.2f}")
    
    # No limit - show all plot data
    # if len(plot_df) > 100:
    #     print(f"Limiting plot data from {len(plot_df)} to 100 points for testing")
    #     plot_df = plot_df.head(100)
    
    # For true percentile stretching, we need to pre-calculate colors
    # since HexagonLayer's color_range doesn't support percentile stretching
    if variable in plot_df.columns:
        values = plot_df[variable].dropna()
        if len(values) > 0:
            q2 = values.quantile(0.02)
            q98 = values.quantile(0.98)
            print(f"HEXAGON: Percentile range for {variable}: {q2:.3f} to {q98:.3f}")
            
            def get_percentile_color(value):
                """Convert value to color using percentile stretching"""
                if pd.isna(value):
                    return [128, 128, 128, 255]  # Gray for missing values
                
                # Clamp value to percentile range and normalize
                clamped_value = max(q2, min(q98, value))
                if q98 == q2:
                    normalized = 0.5
                else:
                    normalized = (clamped_value - q2) / (q98 - q2)
                
                # Invert for diversity metrics (high diversity = green)
                if 'diversity' in variable.lower():
                    normalized = 1 - normalized
                
                # Green to Yellow to Red color ramp
                if normalized <= 0.5:
                    # Green to Yellow
                    ratio = normalized * 2
                    r = int(255 * ratio)
                    g = 255
                    b = 0
                else:
                    # Yellow to Red
                    ratio = (normalized - 0.5) * 2
                    r = 255
                    g = int(255 * (1 - ratio))
                    b = 0
                
                return [r, g, b, 255]
            
            # Pre-calculate colors for each data point
            plot_df = plot_df.copy()
            plot_df['color'] = plot_df[variable].apply(get_percentile_color)
            print(f"HEXAGON: Pre-calculated colors using percentile stretching")
        else:
            plot_df = plot_df.copy()
            plot_df['color'] = [[255, 0, 0, 255] for _ in range(len(plot_df))]
    else:
        plot_df = plot_df.copy()
        plot_df['color'] = [[255, 0, 0, 255] for _ in range(len(plot_df))]
    
    # Use HexagonLayer for density visualization with percentile-stretched colors
    # We'll use a custom color function that applies percentile stretching
    def get_hexagon_color_weight(row):
        """Custom color weight function that applies percentile stretching"""
        value = row[variable]
        if pd.isna(value):
            return 0.5  # Neutral value for missing data
        
        # Apply percentile stretching
        clamped_value = max(q2, min(q98, value))
        if q98 == q2:
            return 0.5
        else:
            normalized = (clamped_value - q2) / (q98 - q2)
            # Invert for diversity metrics
            if 'diversity' in variable.lower():
                normalized = 1 - normalized
            return normalized
    
    # Add color weight column
    plot_df = plot_df.copy()
    plot_df['color_weight'] = plot_df.apply(get_hexagon_color_weight, axis=1)
    
    hexagon_layer = pdk.Layer(
        'HexagonLayer',
        data=plot_df,
        get_position=['longitude', 'latitude'],
        get_color_weight='color_weight',  # Use percentile-stretched color weights
        color_aggregation='MEAN',  # Average the color weights in each hexagon
        color_range=[[0, 255, 0], [255, 255, 0], [255, 0, 0]],  # Green to Yellow to Red
        auto_highlight=True,
        elevation_scale=100,
        pickable=True,
        elevation_range=[0, 3000],
        extruded=True,
        coverage=1,
        radius=1000,  # Hexagon radius in meters
        id=f'plot-hexagon-layer-{variable}'
    )
    
    print(f"HEXAGON: Created hexagon layer with {len(plot_df)} points for variable: {variable}")
    print(f"HEXAGON: Using percentile-stretched color weights (2nd-98th percentiles)")
    print(f"HEXAGON: Color range: Green -> Yellow -> Red")
    print(f"HEXAGON: Hexagon radius: 1000 meters")
    
    # Debug: Check the actual data being sent to the layer
    if len(plot_df) > 0:
        print(f"HEXAGON: Data range for {variable}: {plot_df[variable].min():.3f} to {plot_df[variable].max():.3f}")
        print(f"HEXAGON: Sample data points: {len(plot_df)} total points")
    
    return hexagon_layer

def clean_geometry_for_export(geometry_json):
    """Clean geometry to only include polygons and multipolygons"""
    if not geometry_json:
        return None
    
    try:
        geom = json.loads(geometry_json) if isinstance(geometry_json, str) else geometry_json
        
        # Only keep polygons and multipolygons
        if geom.get('type') in ['Polygon', 'MultiPolygon']:
            return geom
        elif geom.get('type') == 'GeometryCollection':
            # Extract only polygons and multipolygons from geometry collection
            clean_geometries = []
            for sub_geom in geom.get('geometries', []):
                if sub_geom.get('type') in ['Polygon', 'MultiPolygon']:
                    clean_geometries.append(sub_geom)
            
            if len(clean_geometries) == 1:
                return clean_geometries[0]
            elif len(clean_geometries) > 1:
                return {
                    "type": "MultiPolygon",
                    "coordinates": [g['coordinates'] for g in clean_geometries if g['type'] == 'Polygon'] +
                                  [coord for g in clean_geometries if g['type'] == 'MultiPolygon' for coord in g['coordinates']]
                }
            else:
                return None  # No valid geometries found
        else:
            return None  # Not a polygon, multipolygon, or geometry collection
    except Exception as e:
        print(f"Error cleaning geometry: {e}")
        return None

def export_facility_data_as_geojson():
    """Export facility data as GeoJSON"""
    try:
        # Query the stat_supply_shed table
        query = f"""
        SELECT 
            * EXCEPT(geo),
            ST_ASGEOJSON(geo) as geometry
        FROM `{PROJECT_ID}.{DATASET_ID}.stat_supply_shed`
        WHERE geo IS NOT NULL
        """
        
        job = client.query(query)
        results = job.result()
        
        # Convert to GeoJSON format
        features = []
        for row in results:
            # Clean the geometry to only include polygons and multipolygons
            clean_geometry = clean_geometry_for_export(row.geometry)
            if clean_geometry:  # Only include features with valid geometry
                feature = {
                    "type": "Feature",
                    "properties": {
                        "facility_id": row.facility_id,
                        "company_name": row.company_name,
                        "country": row.country
                    },
                    "geometry": clean_geometry
                }
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return json.dumps(geojson, cls=SafeJSONEncoder)
    except Exception as e:
        print(f"Error exporting facility data: {e}")
        return None

def export_plot_data_as_geoparquet():
    """Export plot data as GeoParquet"""
    try:
        # Query the stat_plot table
        query = f"""
        SELECT 
            * EXCEPT(geo),
            luc_tco2ehayear + nonluc_tco2ehayear as total_tco2ehayear,
            luc_tco2ehayear + nonluc_tco2ehayear as total_tco2ehayear,
            ST_ASGEOJSON(geo) as geometry
        FROM `{PROJECT_ID}.{DATASET_ID}.stat_plot`
        WHERE geo IS NOT NULL
        """
        
        job = client.query(query)
        results = job.result()
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in results])
        
        # Clean the dataframe
        df = df.replace([np.nan, np.inf, -np.inf], None)
        df = df.where(pd.notnull(df), None)
        
        # Clean geometries and convert to shapely objects
        geometries = []
        valid_rows = []
        
        for idx, row in df.iterrows():
            clean_geometry = clean_geometry_for_export(row.geometry)
            if clean_geometry:
                try:
                    geom = shape(clean_geometry)
                    geometries.append(geom)
                    valid_rows.append(row)
                except Exception as e:
                    print(f"Error converting geometry for row {idx}: {e}")
                    continue
        
        if not geometries:
            print("No valid geometries found for export")
            return None
        
        # Create GeoDataFrame with valid geometries
        gdf = gpd.GeoDataFrame(valid_rows, geometry=geometries, crs='EPSG:4326')
        
        buffer = io.BytesIO()
        gdf.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        return buffer.getvalue()
    except Exception as e:
        print(f"Error exporting plot data: {e}")
        return None

def export_detail_map_data_as_geojson(collection_id):
    """Export detail map data as GeoJSON"""
    try:
        # Query the collection-specific plot table
        query = f"""
        SELECT 
            * EXCEPT(geo),
            luc_tco2eyear + nonluc_tco2eyear as total_tco2eyear,
            luc_tco2ehayear + nonluc_tco2ehayear as total_tco2ehayear,
            ST_ASGEOJSON(geo) as geometry
        FROM `{PROJECT_ID}.{DATASET_ID}.{collection_id}_plot`
        WHERE geo IS NOT NULL
        """
        
        job = client.query(query)
        results = job.result()
        
        # Convert to GeoJSON format
        features = []
        for row in results:
            # Clean the geometry to only include polygons and multipolygons
            clean_geometry = clean_geometry_for_export(row.geometry)
            if clean_geometry:  # Only include features with valid geometry
                feature = {
                    "type": "Feature",
                    "properties": {
                        "noncompliance_area_ha": row.noncompliance_area_ha,
                        "noncompliance_area_perc": row.noncompliance_area_perc,
                        "luc_tco2eyear": row.luc_tco2eyear,
                        "luc_tco2ehayear": row.luc_tco2ehayear,
                        "nonluc_tco2eyear": row.nonluc_tco2eyear,
                        "nonluc_tco2ehayear": row.nonluc_tco2ehayear,
                        "diversity_score": row.diversity_score,
                        "precipitation_pet_ratio": row.precipitation_pet_ratio,
                        "soil_moisture_percentile": row.soil_moisture_percentile,
                        "evapotranspiration_anomaly": row.evapotranspiration_anomaly,
                        "water_stress_index": row.water_stress_index,
                        "area_ha": row.area_ha,
                        "total_tco2eyear": row.total_tco2eyear,
                        "total_tco2ehayear": row.total_tco2ehayear
                    },
                    "geometry": clean_geometry
                }
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return json.dumps(geojson, cls=SafeJSONEncoder)
    except Exception as e:
        print(f"Error exporting detail map data: {e}")
        return None

def create_default_detail_map():
    """Create a default detail map with sample data from the specified default collection"""
    try:
        # Use the specified default collection ID
        default_collection_id = "0x706310441550789c3dc429e3098592f3702e07e8b36cae7c06e96daf1a85a65d"
        print(f"Loading default collection: {default_collection_id}")
            
        # Query plot data for the first collection
        plot_query = f"""
            SELECT 
                noncompliance_area_ha,
                noncompliance_area_perc,
                luc_tco2eyear,
                luc_tco2ehayear,
                nonluc_tco2eyear,
                nonluc_tco2ehayear,
                diversity_score,
                precipitation_pet_ratio,
                soil_moisture_percentile,
                evapotranspiration_anomaly,
                water_stress_index,
                area_ha,
                luc_tco2eyear + nonluc_tco2eyear as total_tco2eyear,
                luc_tco2ehayear + nonluc_tco2ehayear as total_tco2ehayear,
                ST_ASGEOJSON(geo) as geometry
            FROM `{PROJECT_ID}.{DATASET_ID}.{default_collection_id}_plot`
            WHERE geo IS NOT NULL 
            """
            
            # Query supply shed data for the first collection
        supply_shed_query = f"""
            SELECT 
                uuid,
                ST_ASGEOJSON(geo) as geometry
            FROM `{PROJECT_ID}.{DATASET_ID}.{default_collection_id}_supply_shed`
            WHERE geo IS NOT NULL
            """
            
        # Execute queries
        plot_job = client.query(plot_query)
        supply_shed_job = client.query(supply_shed_query)
        
        plot_results = plot_job.result()
        supply_shed_results = supply_shed_job.result()
        
        # Convert to DataFrames
        plot_df = pd.DataFrame([dict(row) for row in plot_results])
        supply_shed_df = pd.DataFrame([dict(row) for row in supply_shed_results])
        
        # Clean the dataframes to ensure JSON serializability
        plot_df = plot_df.replace([np.nan, np.inf, -np.inf], None)
        supply_shed_df = supply_shed_df.replace([np.nan, np.inf, -np.inf], None)
        
        plot_df = plot_df.where(pd.notnull(plot_df), None)
        supply_shed_df = supply_shed_df.where(pd.notnull(supply_shed_df), None)
        
        print(f"Loaded default: {len(plot_df)} plot records and {len(supply_shed_df)} supply shed records")
        
        return [plot_df, supply_shed_df]
    except Exception as e:
        print(f"Error creating default detail map: {e}")
        return None

def create_detail_map(plot_df, supply_shed_df, color_field='total_tco2ehayear'):
    """Create a detail map showing plot and supply shed data using GeoJSON layers"""
    layers = []
    
    # Add supply shed layer (outline only, no fill)
    if not supply_shed_df.empty:
        # Convert geometry strings to GeoJSON features
        supply_shed_features = []
        
        for i, (_, row) in enumerate(supply_shed_df.iterrows()):
            try:
                geom = json.loads(row['geometry'])
                properties = {
                    "uuid": row.get('uuid', f'supply_shed_{i}'),
                    "type": "supply_shed"
                }
                
                feature = {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": properties
                }
                supply_shed_features.append(feature)
            except Exception as e:
                continue
        
        if supply_shed_features:
            supply_shed_geojson = {
                "type": "FeatureCollection",
                "features": supply_shed_features
            }
            
            supply_shed_layer = pdk.Layer(
                "GeoJsonLayer",
                data=supply_shed_geojson,
                get_fill_color=[0, 0, 0, 0],  # No fill
                get_line_color=[255, 0, 255, 255],  # Magenta outline
                line_width_min_pixels=2,
                line_width_max_pixels=5,
                pickable=False,
                auto_highlight=False
            )
            layers.append(supply_shed_layer)
    
    # Add plot layer using GeoJSON with same color mapping as main map
    if not plot_df.empty:
        # Get the same color scale as the main map
        plot_data = plot_df.copy()
        
        # Clean NaN values from the dataframe (but preserve actual data)
        numeric_columns = plot_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in plot_data.columns:
                # Only fill NaN values, don't replace existing data
                plot_data[col] = plot_data[col].fillna(0)
        plot_data = plot_data.replace([np.inf, -np.inf], 0)
        
        # Use the same quantile-based color mapping as the main map
    def get_color_ramp(value, data_series, field_name):
        """Convert value to appropriate color ramp based on field type"""
        if pd.isna(value) or value is None:
            return [128, 128, 128, 255]  # Gray for missing values (fully opaque)
        
        # Use 5th and 95th percentiles for more aggressive stretching
        q2 = data_series.quantile(0.05)
        q98 = data_series.quantile(0.95)
        
        if q98 == q2:
            normalized = 0.5
        else:
            # Clamp value to quantile range and normalize
            clamped_value = max(q2, min(q98, value))
            normalized = (clamped_value - q2) / (q98 - q2)
        
        # For diversity_score, invert the color ramp (high = good = green)
        if 'diversity' in field_name.lower():
            normalized = 1 - normalized  # Invert for diversity
        
        # Bright green-to-red color ramp: green -> yellow -> orange -> red
        if normalized < 0.33:
            # Green to yellow
            ratio = normalized / 0.33
            r = int(0 + (255 - 0) * ratio)
            g = 255
            b = 0
        elif normalized < 0.66:
            # Yellow to orange
            ratio = (normalized - 0.33) / 0.33
            r = 255
            g = int(255 + (165 - 255) * ratio)
            b = 0
        else:
            # Orange to red
            ratio = (normalized - 0.66) / 0.34
            r = 255
            g = int(165 + (0 - 165) * ratio)
            b = 0
        
        return [r, g, b, 255]  # Fully opaque for maximum brightness
    
    # Use the specified color field for coloring
    if color_field in plot_data.columns:
        color_column = color_field
    else:
        color_column = None
    
    if color_column:
        plot_data['color'] = plot_data[color_column].apply(
            lambda x: get_color_ramp(x, plot_data[color_column], color_field)
        )
        
    else:
        # Fallback to gray if no color column available
        plot_data['color'] = [[128, 128, 128, 255] for _ in range(len(plot_data))]  # Gray for all rows
    
    # Convert geometry strings to GeoJSON features
    plot_features = []
    
    for i, (_, row) in enumerate(plot_data.iterrows()):
        try:
            geom = json.loads(row['geometry'])
            properties = {
                "plot_id": row.get('plot_id', f'plot_{i}'),
                "area_ha": row.get('area_ha', 0),
                "luc_tco2eyear": row.get('luc_tco2eyear', 0),
                "nonluc_tco2eyear": row.get('nonluc_tco2eyear', 0),
                "total_tco2ehayear": row.get('total_tco2ehayear', 0),
                "noncompliance_area_ha": row.get('noncompliance_area_ha', 0),
                "noncompliance_area_perc": row.get('noncompliance_area_perc', 0),
                "diversity_score": row.get('diversity_score', 0),
                "water_stress_index": row.get('water_stress_index', 0),
                "color": row['color']
            }
            
            
            # Helper function to clean NaN values
            def clean_value(value):
                if pd.isna(value) or value is None:
                    return 0
                return value
            
            # Flatten properties for tooltip compatibility
            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": properties,
                # Add properties at root level for tooltip access (cleaned)
                "plot_id": str(properties["plot_id"]) if properties["plot_id"] is not None else "unknown",
                "area_ha": clean_value(properties["area_ha"]),
                "noncompliance_area_ha": clean_value(properties["noncompliance_area_ha"]),
                "noncompliance_area_perc": clean_value(row.get('noncompliance_area_perc', 0)),
                "total_tco2ehayear": clean_value(properties["total_tco2ehayear"]),
                "diversity_score": clean_value(properties["diversity_score"]),
                "water_stress_index": clean_value(properties["water_stress_index"])
            }
            plot_features.append(feature)
        except Exception as e:
            continue
    
    if plot_features:
        # Create GeoJSON layer for plots
        plot_geojson = {
            "type": "FeatureCollection",
            "features": plot_features
        }
        
        
        # Create plot layer using GeoJsonLayer
        plot_layer = pdk.Layer(
            "GeoJsonLayer",
            data=plot_geojson,
            get_fill_color="properties.color",
            get_line_color=[255, 255, 255, 255],  # White outline with transparency
            line_width_min_pixels=12,  # Thick outline
            line_width_max_pixels=20,  # Very thick outline
            pickable=True,
            auto_highlight=True,
            extruded=True,
            get_elevation="properties.noncompliance_area_perc",
            elevation_scale=10000,
            elevation_range=[0, 3000]
        )
        layers.append(plot_layer)
    
    # Create view state centered on the data
    if not plot_df.empty or not supply_shed_df.empty:
        # Calculate bounds from all geometries
        all_lats = []
        all_lons = []
        
        for _, row in pd.concat([plot_df, supply_shed_df], ignore_index=True).iterrows():
            try:
                geom = json.loads(row['geometry'])
                if geom['type'] == 'Point':
                    all_lons.append(geom['coordinates'][0])
                    all_lats.append(geom['coordinates'][1])
                elif geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]
                    for coord in coords:
                        all_lons.append(coord[0])
                        all_lats.append(coord[1])
            except:
                continue
        
        if all_lats and all_lons:
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)
            
            # Calculate zoom based on data extent
            lat_range = max(all_lats) - min(all_lats)
            lon_range = max(all_lons) - min(all_lons)
            max_range = max(lat_range, lon_range)
            
            if max_range > 1:
                zoom = 8
            elif max_range > 0.1:
                zoom = 10
            else:
                zoom = 12
        else:
            center_lat, center_lon, zoom = 0, 0, 2
    else:
        center_lat, center_lon, zoom = 0, 0, 2
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        bearing=0,  # 45-degree bearing for better 3D view
        pitch=60    # 60-degree pitch for better elevation visibility
    )
    
    # Create deck configuration with simpler tooltip
    tooltip_config = {
        "html": "Plot ID: {properties.plot_id}<br/>"
               "Area: {properties.area_ha} ha<br/>"
               "Deforestation Area (ha): {properties.noncompliance_area_ha} ha<br/>"
               "Deforestation Area (%): {properties.noncompliance_area_perc} ha<br/>"
               "CO2: {properties.total_tco2ehayear} tCO2e/ha/year<br/>"
               "Diversity: {properties.diversity_score}<br/>"
               "Water Stress: {properties.water_stress_index}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    
    
    try:
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v11",  # Latest satellite style
            api_keys={"mapbox": mapbox_api_token},
            height=500,
            width="100%",
            tooltip=True
        )
        
        # Convert to JSON for dash_deck.DeckGL component with safe encoding for NaN values
        deck_json = deck.to_json()
        
        # Parse and re-encode with safe encoder to catch any remaining NaN values
        try:
            deck_data = json.loads(deck_json)
            return json.dumps(deck_data, cls=SafeJSONEncoder)
        except:
            # If parsing fails, return the original JSON
            return deck_json
    except Exception as e:
        print(f"Error creating detail map deck: {e}")
        # Return empty map if there's an error
        return json.dumps({"layers": []}, cls=SafeJSONEncoder)

# Callback to update hovered facility store from map click
@app.callback(
    Output('hovered-facility', 'data', allow_duplicate=True),
    Input('deck-map', 'clickInfo'),
    prevent_initial_call=True
)
def update_hovered_facility_from_map(map_click_data):
    """Update hovered facility store when map is clicked"""
    if map_click_data and 'object' in map_click_data and map_click_data['object'] is not None:
        obj = map_click_data['object']
        # Create the same facility label format as used in the chart
        facility_id = obj.get('facility_id', '')
        company_name = obj.get('company_name', '')
        
        # Clean the values
        if isinstance(facility_id, str):
            facility_id = facility_id.strip()
        if isinstance(company_name, str):
            company_name = company_name.strip()
        
        # Check if values are meaningful
        def is_meaningful(value):
            if pd.isna(value):
                return False
            if isinstance(value, str) and value.strip().upper() in ['NA', 'N/A', 'NULL', '']:
                return False
            return True
        
        facility_meaningful = is_meaningful(facility_id)
        company_meaningful = is_meaningful(company_name)
        
        if facility_meaningful and company_meaningful:
            return f"{facility_id} - {company_name}"
        elif facility_meaningful:
            return str(facility_id)
        elif company_meaningful:
            return str(company_name)
    return None

# Callback to update tooltip based on layer toggle
@app.callback(
    Output('deck-map', 'tooltip'),
    Input('main-map-layer-toggle', 'value'),
    Input('y-axis-dropdown', 'value'),
    prevent_initial_call=True
)
def update_map_tooltip(layer_toggle, variable):
    """Update tooltip based on layer type"""
    if layer_toggle:  # Plot layer (HexagonLayer)
        return {
            "html": f"<b>Plot Density</b><br/>"
                   f"<b>Variable ({variable}):</b> {{elevationValue}}<br/>"
                   f"<b>Plot Count:</b> {{count}}<br/>"
                   f"<b>Position:</b> {{x}}, {{y}}<br/>"
                   f"<b>Hexagon ID:</b> {{hexagon_id}}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px"
            }
        }
    else:  # Facility layer
        return {
            "html": "<b>Location ID:</b> {facility_id}<br/>"
                   "<b>Company:</b> {company_name}<br/>"
                   "<b>Country:</b> {country}<br/>"
                   "<b>Commodity Area (ha):</b> {area_ha_commodity} ha<br/>"
                   "<b>Noncompliance Area (%):</b> {noncompliance_area_perc}<br/>"
                   "<b>Total Emissions (tCO2e/ha/year):</b> {total_tco2ehayear}<br/>"
                   "<b>Diversity Score:</b> {diversity_score}<br/>"
                   "<b>Water Stress:</b> {water_stress_index}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px"
            }
        }

# Callback to update map based on highlighted facility store
@app.callback(
    Output('deck-map', 'data'),
    [Input('y-axis-dropdown', 'value'),
     Input('highlighted-facility', 'data'),
     Input('main-map-layer-toggle', 'value')],
    prevent_initial_call=True
)
def update_deck_map(variable, highlighted_facility, layer_toggle):
    """Update the deck.gl map data and view state based on highlighted facility store"""
    
    print(f"TOGGLE: Callback triggered - layer_toggle: {layer_toggle}, variable: {variable}")
    print(f"TOGGLE: All inputs - variable: {variable}, highlighted_facility: {highlighted_facility}, layer_toggle: {layer_toggle}")
    
    # Add a small delay to make loading visible
    import time
    time.sleep(0.5)  # 500ms delay to show loading spinner
    
    # If there's a highlighted facility, find its coordinates for custom view state
    custom_view_state = None
    if highlighted_facility:
        # Find the coordinates of the highlighted facility
        filtered_df = df.copy()
        
        # Create facility_company labels to match
        def create_facility_label(row):
            facility_id = row.get('facility_id', 'Unknown')
            company_name = row.get('company_name', 'Unknown')
            
            if isinstance(facility_id, str):
                facility_id = facility_id.strip()
            if isinstance(company_name, str):
                company_name = company_name.strip()
            
            def is_meaningful(value):
                if pd.isna(value):
                    return False
                if isinstance(value, str) and value.strip().upper() in ['NA', 'N/A', 'NULL', '']:
                    return False
                return True
            
            facility_meaningful = is_meaningful(facility_id)
            company_meaningful = is_meaningful(company_name)
            
            if facility_meaningful and company_meaningful:
                return f"{facility_id} - {company_name}"
            elif facility_meaningful:
                return str(facility_id)
            elif company_meaningful:
                return str(company_name)
            else:
                return "Unknown"
        
        filtered_df['facility_label'] = filtered_df.apply(create_facility_label, axis=1)
        matching_facility = filtered_df[filtered_df['facility_label'] == highlighted_facility]
        
        if not matching_facility.empty:
            lat = matching_facility['latitude'].iloc[0]
            lon = matching_facility['longitude'].iloc[0]
            
            # Create a zoomed view state
            custom_view_state = {
                'latitude': lat,
                'longitude': lon,
                'zoom': 12,  # Zoom level - adjust as needed
                'pitch': 0,  # 2D view
                'bearing': 0
            }
    
    # Create map based on layer type (True = Plots, False = Facilities)
    if layer_toggle:
        print(f"TOGGLE: Creating plots layer with variable: {variable}")
        # Load plot data when needed
        global plot_df
        if plot_df is None:
            print("TOGGLE: Loading plot data for first time...")
            plot_df = fetch_plot_hexagon_data()
            print(f"TOGGLE: Loaded plot data, shape: {plot_df.shape if not plot_df.empty else 'empty'}")
        
        if not plot_df.empty:
            print(f"TOGGLE: Plot data columns: {list(plot_df.columns)}")
            hexagon_layer = create_plot_hexagon_layer(plot_df, variable)
            print(f"TOGGLE: Created hexagon layer: {type(hexagon_layer)}")
            
            # Calculate proper view state for plot data
            if not plot_df.empty:
                # Get bounds of the plot data
                min_lat = plot_df['latitude'].min()
                max_lat = plot_df['latitude'].max()
                min_lon = plot_df['longitude'].min()
                max_lon = plot_df['longitude'].max()
                
                print(f"TOGGLE: Plot data bounds - Lat: {min_lat:.2f} to {max_lat:.2f}, Lon: {min_lon:.2f} to {max_lon:.2f}")
                
                # Calculate center
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                
                print(f"TOGGLE: Plot data center - Lat: {center_lat:.2f}, Lon: {center_lon:.2f}")
                
                # Calculate zoom level based on data spread
                lat_range = max_lat - min_lat
                lon_range = max_lon - min_lon
                max_range = max(lat_range, lon_range)
                
                # Adjust zoom based on data spread
                if max_range > 10:
                    zoom_level = 3
                elif max_range > 5:
                    zoom_level = 4
                elif max_range > 2:
                    zoom_level = 5
                else:
                    zoom_level = 6
                
                plot_view_state = {
                    'latitude': center_lat,
                    'longitude': center_lon,
                    'zoom': zoom_level,
                    'pitch': 60,  # 60-degree pitch for 3D view like detail map
                    'bearing': 45  # 45-degree bearing for better 3D view like detail map
                }
            else:
                plot_view_state = {
                    'latitude': 0,
                    'longitude': 0,
                    'zoom': 2,
                    'pitch': 60,  # 60-degree pitch for 3D view like detail map
                    'bearing': 45  # 45-degree bearing for better 3D view like detail map
                }
            
            # Create deck data using pydeck like the facility layer
            import time
            
            deck = pdk.Deck(
                layers=[hexagon_layer],
                initial_view_state=plot_view_state,
                map_style='mapbox://styles/mapbox/dark-v11',
                api_keys={"mapbox": mapbox_api_token},
                height=500,
                width="100%",
                tooltip=True
            )
            print(f"TOGGLE: Returning pydeck object with {len([hexagon_layer])} layers")
            return deck.to_json()
        else:
            print("TOGGLE: No plot data found, returning empty map")
            # Fallback to empty map if no plot data
            return json.dumps({'layers': []}, cls=SafeJSONEncoder)
    else:
        print("TOGGLE: Creating facilities layer")
        # Default to facility layer
        facility_data = create_deck_map(None, 'dark', variable, '2d', custom_view_state, highlighted_facility)
        return facility_data

# Callback for main map export
@app.callback(
    [Output("main-map-download", "data"),
     Output("main-map-export-spinner", "style")],
    [Input("main-map-export-btn", "n_clicks"),
     Input("main-map-layer-toggle", "value")],
    prevent_initial_call=True
)
def export_main_map_data(n_clicks, layer_toggle):
    """Export main map data based on current layer"""
    if n_clicks is None:
        return None, {"display": "none"}
    
    # Show spinner immediately
    spinner_style = {
        "marginLeft": "10px",
        "width": "16px",
        "height": "16px",
        "border": "2px solid #f3f3f3",
        "borderTop": "2px solid #007bff",
        "borderRadius": "50%",
        "animation": "spin 1s linear infinite",
        "display": "block"
    }
    
    # Add a delay to make the export process visible
    import time
    time.sleep(3.0)  # 3 second delay to show export is happening
    
    if layer_toggle:  # Plot layer
        # Export plot data as GeoParquet
        data = export_plot_data_as_geoparquet()
        if data:
            return dict(content=data, filename="plot_data.parquet"), {"display": "none"}
    else:  # Facility layer
        # Export facility data as GeoJSON
        data = export_facility_data_as_geojson()
        if data:
            return dict(content=data, filename="facility_data.geojson"), {"display": "none"}
    
    return None, {"display": "none"}

# Callback for detail map export
@app.callback(
    [Output("detail-map-download", "data"),
     Output("detail-map-export-spinner", "style")],
    [Input("detail-map-export-btn", "n_clicks")],
    [State("detail-map-data", "data")],
    prevent_initial_call=True
)
def export_detail_map_data(n_clicks, stored_data):
    """Export detail map data as GeoJSON"""
    if n_clicks is None or not stored_data:
        return None, {"display": "none"}
    
    # Show spinner immediately
    spinner_style = {
        "marginLeft": "10px",
        "width": "16px",
        "height": "16px",
        "border": "2px solid #f3f3f3",
        "borderTop": "2px solid #007bff",
        "borderRadius": "50%",
        "animation": "spin 1s linear infinite",
        "display": "block"
    }
    
    # Add a delay to make the export process visible
    import time
    time.sleep(3.0)  # 3 second delay to show export is happening
    
    # Get the current collection ID from stored data
    collection_id = stored_data.get('collection_id')
    if not collection_id:
        print("No collection_id found in stored data, using default")
        collection_id = "0x706310441550789c3dc429e3098592f3702e07e8b36cae7c06e96daf1a85a65d"
    
    print(f"Exporting detail map data for collection: {collection_id}")
    data = export_detail_map_data_as_geojson(collection_id)
    if data:
        return dict(content=data, filename=f"detail_map_data_{collection_id}.geojson"), {"display": "none"}
    
    return None, {"display": "none"}

# Callback to update chart based on highlighted facility store
@app.callback(
    Output('main-chart', 'figure'),
    [Input('y-axis-dropdown', 'value'),
     Input('highlighted-facility', 'data')],
    prevent_initial_call=True
)
def update_main_chart(chart_var, highlighted_facility):
    """Update single bar chart based on highlighted facility store"""
    
    # Handle None values from dropdown (initial load)
    if chart_var is None:
        chart_var = 'total_tco2ehayear'
    
    # Always show all data (no filtering)
    filtered_df = df.copy()
    
    # Create chart with selected variable and highlighting
    fig = create_chart(filtered_df, chart_var, EPOCH_COLORS['primary'], highlighted_facility)
    
    return fig

def create_facility_metadata_table(facility_data):
    """Create a metadata table for the selected facility"""
    
    # Define the fields to display and their labels
    metadata_fields = {
        'facility_id': 'Facility ID',
        'company_name': 'Company Name',
        'country': 'Country',
        'area_ha_commodity': 'Commodity Area (ha)',
        'area_ha_smallholder': 'Smallholder Area (ha)',
        'area_ha_estate': 'Estate Area (ha)',
        'area_ha_supply_shed': 'Supply Shed Area (ha)',
        'area_ha_forest': 'Natural Forest Area (ha)',
        'noncompliance_area_rate': 'Noncompliance Rate',
        'noncompliance_area_perc': 'Noncompliance Area (%)',
        'luc_tco2eyear': 'LUC Emissions (tCO2e/year)',
        'luc_tco2ehayear': 'LUC Emissions (tCO2e/ha/year)',
        'nonluc_tco2eyear': 'Non-LUC Emissions (tCO2e/year)',
        'nonluc_tco2ehayear': 'Non-LUC Emissions (tCO2e/ha/year)',
        'total_tco2eyear': 'Total Emissions (tCO2e/year)',
        'total_tco2ehayear': 'Total Emissions (tCO2e/ha/year)',
        'diversity_score': 'Diversity Score',
        'water_stress_index': 'Water Stress Index',
        'precipitation_pet_ratio': 'Precipitation-PET Ratio',
        'soil_moisture_percentile': 'Soil Moisture Percentile',
        'evapotranspiration_anomaly': 'Evapotranspiration Anomaly',
        'estate_smallholder_ratio': 'Estate/Smallholder Ratio'
    }
    
    # Create table rows
    table_rows = []
    for field, label in metadata_fields.items():
        if field in facility_data:
            value = facility_data[field]
            # Format numeric values
            if pd.isna(value):
                display_value = "N/A"
            elif isinstance(value, (int, float)):
                if field.endswith('_ha') or field.endswith('_year'):
                    display_value = f"{value:,.0f}"
                elif field.endswith('_rate') or field.endswith('_perc') or field.endswith('_ratio'):
                    display_value = f"{value:.2f}"
                else:
                    display_value = f"{value:.3f}"
            else:
                display_value = str(value)
            
            table_rows.append(
                html.Tr([
                    html.Td(label, style={'fontWeight': 'bold', 'width': '40%'}),
                    html.Td(display_value, style={'width': '60%'})
                ])
            )
    
    # Create the table with proper tbody wrapper
    metadata_table = dbc.Table(
        [
            html.Tbody(table_rows)
        ],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
        style={'backgroundColor': EPOCH_COLORS['secondary'], 'color': 'white'}
    )
    
    return metadata_table

def create_chart(filtered_df, y_column, chart_color, highlighted_facility=None):
    """Create a single bar chart with facility_id + company name on x-axis and specified y-column"""
    
    # Create a combined identifier for x-axis
    filtered_df = filtered_df.copy()
    
    # Helper function to check if a value is meaningful (not NA, NaN, or empty string)
    def is_meaningful(value):
        if pd.isna(value):
            return False
        if isinstance(value, str) and value.strip().upper() in ['NA', 'N/A', 'NULL', '']:
            return False
        return True
    
    # Debug: Print available columns
    print(f"CHART: Available columns in filtered_df: {list(filtered_df.columns)}")
    
    # Create facility_company field with better handling
    def create_facility_label(row):
        # Check if columns exist and get values safely
        facility_id = row.get('facility_id', 'Unknown')
        company_name = row.get('company_name', 'Unknown')
        
        # Clean the values
        if isinstance(facility_id, str):
            facility_id = facility_id.strip()
        if isinstance(company_name, str):
            company_name = company_name.strip()
        
        # Check if values are meaningful
        facility_meaningful = is_meaningful(facility_id)
        company_meaningful = is_meaningful(company_name)
        
        if facility_meaningful and company_meaningful:
            return f"{facility_id} - {company_name}"
        elif facility_meaningful:
            return str(facility_id)
        elif company_meaningful:
            return str(company_name)
        else:
            return "Unknown"
    
    filtered_df['facility_company'] = filtered_df.apply(create_facility_label, axis=1)
    
    # Sort by the y-column (highest to lowest)
    df_sorted = filtered_df.sort_values(y_column, ascending=False)
    
    # Create color scale for bars using quantile stretching (same as map)
    def get_color_ramp(value):
        """Convert value to color ramp using quantile stretching with appropriate direction based on field type"""
        # Use quantile stretching (2nd and 98th percentiles) to handle outliers better
        q2 = df_sorted[y_column].quantile(0.02)
        q98 = df_sorted[y_column].quantile(0.98)
        
        if q98 == q2:
            normalized = 0.5
        else:
            # Clamp value to quantile range and normalize
            clamped_value = max(q2, min(q98, value))
            normalized = (clamped_value - q2) / (q98 - q2)
        
        # For diversity_score, invert the color ramp (high = good = green)
        if 'diversity' in y_column.lower():
            normalized = 1 - normalized  # Invert for diversity
        
        # Green to Yellow to Red color ramp
        if normalized <= 0.5:
            # Green to Yellow
            ratio = normalized * 2
            r = int(255 * ratio)
            g = 255
            b = 0
        else:
            # Yellow to Red
            ratio = (normalized - 0.5) * 2
            r = 255
            g = int(255 * (1 - ratio))
            b = 0
        
        return f'rgb({r}, {g}, {b})'
    
    # Apply color scale to bars, with highlighting for hovered facility
    bar_colors = []
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        facility_company = row['facility_company']
        if highlighted_facility and facility_company == highlighted_facility:
            # Highlight the clicked bar with bright yellow
            bar_colors.append('rgb(255, 255, 0)')
        else:
            # Use normal color scale
            bar_colors.append(get_color_ramp(row[y_column]))
    
    # Calculate percentiles for reference lines
    p25 = df_sorted[y_column].quantile(0.25)
    p50 = df_sorted[y_column].quantile(0.50)
    p75 = df_sorted[y_column].quantile(0.75)
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['facility_company'],
        y=df_sorted[y_column],
        name=y_column.replace('_', ' ').title(),
        marker_color=bar_colors,
        customdata=df_sorted['facility_company'],  # Add customdata for cross-filtering
        hovertemplate=f'<b>Facility:</b> %{{x}}<br><b>{y_column.replace("_", " ").title()}:</b> %{{y:.3f}}<extra></extra>'
    ))
    
    # Add percentile reference lines
    fig.add_vline(x=len(df_sorted) * 0.75, line_dash="dash", line_color="#00FF00", 
                  line_width=3, annotation_text="75th percentile", annotation_position="top")
    fig.add_vline(x=len(df_sorted) * 0.50, line_dash="dash", line_color="#FFD700", 
                  line_width=3, annotation_text="50th percentile", annotation_position="top")
    fig.add_vline(x=len(df_sorted) * 0.25, line_dash="dash", line_color="#FF0000", 
                  line_width=3, annotation_text="25th percentile", annotation_position="top")
    fig.add_vline(x=len(df_sorted) * 0.05, line_dash="dash", line_color="#8B0000", 
                  line_width=3, annotation_text="5th percentile", annotation_position="top")
    
    # Update layout
    fig.update_layout(
        title=f"{y_column.replace('_', ' ').title()} by Facility (Highest to Lowest)",
        xaxis_title="Facility ID - Company Name",
        yaxis_title=y_column.replace('_', ' ').title(),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color=EPOCH_COLORS['text_primary']),
        xaxis=dict(
            gridcolor='#f0f0f0',
            showticklabels=False,  # Hide x-axis labels
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            gridcolor='#f0f0f0',
            autorange=True,  # Auto-scale y-axis to fit data
            automargin=True  # Auto-adjust margins
        ),
        # Enable zooming and panning
        dragmode='zoom',
        selectdirection='d'
    )
    
    # Color scale is already applied to bars based on data values
    # No additional highlighting needed since bars are color-coded by value
    
    return fig





# Add callback for detail color dropdown
@app.callback(
    [Output('detail-map', 'data', allow_duplicate=True),
     Output('detail-loading-container', 'style', allow_duplicate=True)],
    Input('detail-color-dropdown', 'value'),
    State('detail-map-data', 'data'),
    prevent_initial_call=True
)
def update_detail_color(color_field, stored_data):
    """Update detail map color based on selected field"""
    
    if not stored_data or not color_field:
        return {'layers': []}, {"display": "none"}
    
    try:
        # Recreate dataframes from stored data
        plot_df = pd.DataFrame(stored_data['plot_df'])
        supply_shed_df = pd.DataFrame(stored_data['supply_shed_df'])
        
        
        # Create new detail map with the selected color field
        result = create_detail_map(plot_df, supply_shed_df, color_field)
        return result, {"display": "none"}
        
    except Exception as e:
        print(f"ERROR in update_detail_color: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'layers': []}, {"display": "none"}

# Add callback for map selection (using double-click to avoid conflict with hover)
@app.callback(
    Output('selected-points', 'data', allow_duplicate=True),
    Input('deck-map', 'clickInfo'),
    State('selected-points', 'data'),
    prevent_initial_call=True
)
def update_selected_points(click_info, current_selection):
    """Update selected points from deck.gl map double-click interaction"""
    if click_info and 'object' in click_info and click_info['object'] is not None:
        # Get the clicked point's index
        clicked_point = click_info['object']
        if 'facility_id' in clicked_point:
            point_id = clicked_point['facility_id']
            # Find the index of this point in our dataframe
            idx = df[df['facility_id'] == point_id].index
            if len(idx) > 0:
                # Toggle selection - if already selected, remove it; if not, add it
                if idx[0] in current_selection:
                    return [i for i in current_selection if i != idx[0]]
                else:
                    return current_selection + [idx[0]]
    return current_selection

# Note: dash_deck.DeckGL doesn't support hoverData property
# Map-chart linking will be handled through click events instead

# Callback to initialize and update metrics
@app.callback(
    [Output('metric-total-facilities', 'children'),
     Output('metric-total-plots', 'children'),
     Output('metric-supply-shed-area', 'children'),
     Output('metric-commodity-area', 'children')],
    [Input('highlighted-facility', 'data')],
    prevent_initial_call=True
)
def update_metrics(highlighted_facility):
    """Update metrics based on highlighted facility or show default values"""
    
    if highlighted_facility:
        # Extract facility ID from the label (format: "FACILITY_ID - COMPANY_NAME")
        facility_id = highlighted_facility
        if ' - ' in highlighted_facility:
            facility_id = highlighted_facility.split(' - ')[0]
        
        # Try different possible column names for facility identification
        facility_data = None
        if 'facility_id' in df.columns:
            facility_data = df[df['facility_id'] == facility_id]
        elif 'facility_company' in df.columns:
            facility_data = df[df['facility_company'] == highlighted_facility]
        elif 'company_name' in df.columns:
            facility_data = df[df['company_name'] == highlighted_facility]
        
        if facility_data is not None and not facility_data.empty:
            facility_row = facility_data.iloc[0]
            return (
                f"{1:,}",  # Single facility
                f"{facility_row.get('commodity_plot_no', 0):,}",
                f"{facility_row.get('area_ha_supply_shed', 0):,.0f}",
                f"{facility_row.get('area_ha_commodity', 0):,.0f}"
            )
    
    # Default metrics from database
    try:
        metrics = get_default_metrics()
        result = (
            f"{metrics['total_facilities']:,}",
            f"{metrics['total_plots']:,}",
            f"{metrics['total_supply_shed_area']:,.0f}",
            f"{metrics['total_commodity_area']:,.0f}"
        )
        return result
    except Exception as e:
        print(f"Error updating metrics: {e}")
        return "0", "0", "0", "0"

if __name__ == '__main__':
    # Get port from environment variable (Cloud Run sets PORT)
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting Supply Shed Visualizer")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Login: william@epoch.blue / ssi123")
    
    app.run(debug=debug, host='0.0.0.0', port=port)  