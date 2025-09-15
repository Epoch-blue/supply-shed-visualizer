# Supply Shed Visualizer

An interactive Plotly Dash application for visualizing supply shed data from BigQuery with linked map and chart displays.

## Features

- **Interactive Map**: Satellite background with geographic data points
- **Linked Charts**: Histograms for key metrics that update based on map selection
- **Cross-filtering**: Hover and select points on the map to highlight corresponding data in charts
- **Real-time Data**: Connects to BigQuery for live data updates

## Metrics Visualized

- Noncompliance Area Rate
- Total tCO2e/ha/year
- Diversity Score
- Water Stress Index

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials:
   - Create a service account key in Google Cloud Console
   - Download the JSON key file
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your key file

3. Create a `.env` file with your BigQuery configuration:
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
BIGQUERY_PROJECT_ID=epoch-geospatial-dev
BIGQUERY_DATASET_ID=1mUTPmnLDbWCneHliVw34sAe1ck1
BIGQUERY_TABLE_ID=stat_supply_shed
```

4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:8050`

## Usage

- **Map Interaction**: Click and drag to select points on the map
- **Chart Updates**: Selected points will be highlighted in the histogram charts
- **Data Summary**: View statistics for selected data points in the sidebar
- **Hover Information**: Hover over map points to see detailed information

## Data Requirements

The BigQuery table should contain:
- `geometry`: Geographic coordinates (will be converted to lat/lon)
- `noncompliance_area_rate`: Float values (0-1)
- `total_tco2ehayear`: Float values
- `diversity_score`: Float values
- `water_stress_index`: Float values

## Development

The app includes sample data generation for development and testing when BigQuery connection is not available.

