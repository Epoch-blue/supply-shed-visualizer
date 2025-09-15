# Supply Shed Visualizer - Cloud Run Deployment Guide

This guide will help you deploy the Supply Shed Visualizer to Google Cloud Run on the `epoch-geospatial-dev` project.

## Prerequisites

1. **Google Cloud CLI** installed and authenticated
2. **Docker** installed (for local testing)
3. **Access** to the `epoch-geospatial-dev` project
4. **Required permissions** in the project

## Quick Deployment

### 1. Setup Service Account
```bash
./setup-service-account.sh
```

This script will:
- Create a service account with proper permissions
- Grant BigQuery and Secret Manager access
- Create a service account key for local development

### 2. Deploy to Cloud Run
```bash
./deploy.sh
```

This script will:
- Build and push the Docker image
- Deploy to Cloud Run
- Configure environment variables
- Set up proper resource limits
- Attempt to create custom domain mapping

### 3. Setup Custom Domain (Optional)
```bash
./setup-domain.sh
```

This script will:
- Create a custom domain mapping for `epoch-supply-shed-viz.app`
- Provide DNS records that need to be added
- Give instructions for domain verification

## Manual Deployment Steps

If you prefer to run the commands manually:

### 1. Set Project
```bash
gcloud config set project epoch-geospatial-dev
```

### 2. Enable APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 3. Configure Existing Service Account
```bash
PROJECT_ID="epoch-geospatial-dev"
SERVICE_ACCOUNT_EMAIL="709579113971-compute@developer.gserviceaccount.com"

# BigQuery permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/bigquery.jobUser"

# Secret Manager permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"
```

### 5. Create Artifact Registry Repository
```bash
# Enable Artifact Registry API
gcloud services enable artifactregistry.googleapis.com

# Create repository
gcloud artifacts repositories create supply-shed-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Supply Shed Visualizer"

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 6. Build and Deploy
```bash
# Build image
gcloud builds submit --tag us-central1-docker.pkg.dev/epoch-geospatial-dev/supply-shed-repo/supply-shed-visualizer

# Deploy to Cloud Run
gcloud run deploy supply-shed-visualizer \
    --image us-central1-docker.pkg.dev/epoch-geospatial-dev/supply-shed-repo/supply-shed-visualizer \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --max-instances 10 \
    --set-env-vars "BIGQUERY_PROJECT_ID=epoch-geospatial-dev" \
    --set-env-vars "BIGQUERY_DATASET_ID=1mUTPmnLDbWCneHliVw34sAe1ck1" \
    --set-env-vars "BIGQUERY_TABLE_ID=stat_supply_shed" \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=epoch-geospatial-dev" \
    --set-env-vars "DEBUG=False"
```

## Environment Variables

The app uses these environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `BIGQUERY_PROJECT_ID` | `epoch-geospatial-dev` | BigQuery project ID |
| `BIGQUERY_DATASET_ID` | `1mUTPmnLDbWCneHliVw34sAe1ck1` | BigQuery dataset ID |
| `BIGQUERY_TABLE_ID` | `stat_supply_shed` | BigQuery table ID |
| `GOOGLE_CLOUD_PROJECT` | `epoch-geospatial-dev` | Google Cloud project ID |
| `DEBUG` | `False` | Debug mode (set to True for development) |

## Local Development

For local development, you'll need the service account key:

1. **Download the service account key** (created by setup script):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
   ```

2. **Create a `.env` file** (optional, for local overrides):
   ```bash
   BIGQUERY_PROJECT_ID=epoch-geospatial-dev
   BIGQUERY_DATASET_ID=1mUTPmnLDbWCneHliVw34sAe1ck1
   BIGQUERY_TABLE_ID=stat_supply_shed
   GOOGLE_CLOUD_PROJECT=epoch-geospatial-dev
   DEBUG=True
   ```

3. **Run locally**:
   ```bash
   python app.py
   ```

## Custom Domain Setup

The deployment script attempts to set up a custom domain: `epoch-supply-shed-viz.app`

### Domain Requirements:
1. **Domain ownership**: You must own the domain
2. **DNS access**: Ability to add DNS records
3. **Domain verification**: May need to verify ownership in Google Cloud Console

### DNS Records:
After running the domain setup, you'll get DNS records like:
```
Type: CNAME
Name: epoch-supply-shed-viz.app
Value: ghs.googlehosted.com
```

### Domain Verification:
1. Go to [Google Cloud Console](https://console.cloud.google.com/run/domains)
2. Select your domain mapping
3. Follow the verification instructions
4. Add the required DNS records to your domain registrar

### Alternative Domains:
You can use any domain you own. Just update the `CUSTOM_DOMAIN` variable in the scripts:
- `epoch-supply-shed-viz.app`
- `supply-shed.epoch.blue`
- `viz.epoch.blue`
- Any other domain you control

## Authentication

The app uses simple username/password authentication:
- **Username**: `william@epoch.blue`
- **Password**: `ssi123`

## Monitoring and Logs

### View Logs
```bash
gcloud run services logs read supply-shed-visualizer --region us-central1
```

### Monitor Performance
```bash
gcloud run services describe supply-shed-visualizer --region us-central1
```

## Updating the Deployment

To update the app:

1. **Make your changes** to the code
2. **Run the deployment script**:
   ```bash
   ./deploy.sh
   ```

The script will automatically:
- Build a new Docker image
- Deploy the updated version
- Keep the same service configuration

## Troubleshooting

### Common Issues

1. **BigQuery Access Denied**
   - Check service account permissions
   - Verify the service account has `BigQuery Data Viewer` and `BigQuery Job User` roles

2. **Secret Manager Access Denied**
   - Ensure the service account has `Secret Manager Secret Accessor` role
   - Verify the `mapbox-api-key` secret exists in Secret Manager

3. **App Won't Start**
   - Check Cloud Run logs: `gcloud run services logs read supply-shed-visualizer --region us-central1`
   - Verify all environment variables are set correctly

4. **Docker Build Fails**
   - Check the Dockerfile syntax
   - Ensure all dependencies are in requirements.txt
   - Verify the build context includes all necessary files

### Getting Help

- Check the Cloud Run console: https://console.cloud.google.com/run
- View logs in the Google Cloud Console
- Use `gcloud` CLI for debugging

## Security Notes

- The app is deployed with `--allow-unauthenticated` for public access
- Authentication is handled at the application level
- Service account credentials are managed by Google Cloud
- No sensitive data is stored in environment variables (uses Secret Manager)

## Cost Optimization

- **CPU**: 2 vCPU (can be reduced to 1 for lower costs)
- **Memory**: 2GB (can be reduced to 1GB if needed)
- **Max Instances**: 10 (adjust based on expected traffic)
- **Timeout**: 3600 seconds (1 hour max request time)
