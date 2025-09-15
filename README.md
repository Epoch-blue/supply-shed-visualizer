# Supply Shed Visualizer

An interactive web application for visualizing supply shed data with advanced mapping capabilities, built with Dash and deployed on Google Cloud Run.

## 🌟 Features

- **Interactive Maps**: PyDeck-powered maps with facility and plot layers
- **Data Visualization**: Hexagon binning for plot data with 3D visualization
- **Export Functionality**: Export data as GeoJSON and GeoParquet formats
- **Authentication**: Simple username/password authentication system
- **Cloud Deployment**: Ready for Google Cloud Run deployment
- **Custom Domains**: Support for professional custom domain URLs

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Project with BigQuery access
- Mapbox API key (stored in Google Secret Manager)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Epoch-blue/supply-shed-visualizer.git
   cd supply-shed-visualizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file with your credentials
   echo "MAPBOX_API_KEY=your_mapbox_key_here" > .env
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the app**: Open http://localhost:8050 in your browser

### Authentication

- **Username**: `william@epoch.blue`
- **Password**: `ssi123`

## 🗺️ Map Features

### Main Map
- **Facility Layer**: Scatter plot of supply shed facilities
- **Plot Layer**: Hexagon-binned plot data with 3D visualization
- **Interactive Tooltips**: Dynamic information based on active layer
- **Export Options**: Download current map data as GeoJSON/GeoParquet

### Detail Map
- **Collection-specific Views**: Detailed plots for selected supply sheds
- **Metadata Display**: Facility information and statistics
- **Export Functionality**: Download collection-specific data

## 📊 Data Sources

- **BigQuery Tables**:
  - `stat_supply_shed`: Supply shed facility data
  - `stat_plot`: Plot-level data with spatial information
  - Collection-specific plot tables: `{collection_hash}_plot`

## 🚀 Deployment

### Google Cloud Run Deployment

1. **Set up service account**:
   ```bash
   ./setup-service-account.sh
   ```

2. **Deploy to Cloud Run**:
   ```bash
   ./deploy.sh
   ```

3. **Set up custom domain** (optional):
   ```bash
   ./setup-domain.sh
   ```

### Manual Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed manual deployment instructions.

## 🛠️ Technology Stack

- **Frontend**: Dash, Dash Bootstrap Components, PyDeck
- **Backend**: Python, Flask
- **Database**: Google BigQuery
- **Maps**: Mapbox, PyDeck
- **Deployment**: Google Cloud Run, Artifact Registry
- **Authentication**: Google Secret Manager

## 📁 Project Structure

```
supply-shed-visualizer/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── deploy.sh             # Cloud Run deployment script
├── setup-service-account.sh  # Service account setup
├── setup-domain.sh       # Custom domain setup
├── DEPLOYMENT.md         # Deployment documentation
├── assets/               # Static assets (CSS, images)
└── public/               # Public files
```

## 🔧 Configuration

### Environment Variables

- `BIGQUERY_PROJECT_ID`: Google Cloud project ID
- `BIGQUERY_DATASET_ID`: BigQuery dataset ID
- `BIGQUERY_TABLE_ID`: Main table ID
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project
- `DEBUG`: Debug mode (True/False)

### BigQuery Configuration

The app connects to BigQuery using the default service account in Cloud Run or a service account key file for local development.

## 📈 Performance

- **Server-side Aggregation**: Plot data is pre-aggregated in BigQuery
- **Optimized Queries**: Spatial queries with proper indexing
- **Caching**: Pre-loaded data for faster map rendering
- **Resource Limits**: 8GB RAM, 4 CPU cores in production

## 🔒 Security

- **Authentication**: Hardcoded credentials (configurable)
- **API Keys**: Stored in Google Secret Manager
- **HTTPS**: Automatic SSL certificates in Cloud Run
- **CORS**: Properly configured for web access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Epoch Blue organization.

## 🆘 Support

For issues and questions:
- Create an issue in this repository
- Contact the development team

## 🔗 Links

- **Repository**: https://github.com/Epoch-blue/supply-shed-visualizer
- **Deployment**: https://epoch-supply-shed-viz.app (when deployed)
- **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

Built with ❤️ by the Epoch Blue team