#!/bin/bash

# Supply Shed Visualizer - Cloud Run Deployment Script
# This script deploys the app to Google Cloud Run

set -e

# Configuration
PROJECT_ID="epoch-geospatial-dev"
SERVICE_NAME="supply-shed-visualizer"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Supply Shed Visualizer to Cloud Run"
echo "   Project: ${PROJECT_ID}"
echo "   Service: ${SERVICE_NAME}"
echo "   Region: ${REGION}"
echo "   Image: ${IMAGE_NAME}"

# Set the project
echo "📋 Setting project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Build and push the Docker image
echo "🏗️  Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --set-env-vars "BIGQUERY_PROJECT_ID=${PROJECT_ID}" \
    --set-env-vars "BIGQUERY_DATASET_ID=1mUTPmnLDbWCneHliVw34sAe1ck1" \
    --set-env-vars "BIGQUERY_TABLE_ID=stat_supply_shed" \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-env-vars "DEBUG=False"

# Optional: Set up custom domain mapping
echo "🌐 Setting up custom domain mapping..."
CUSTOM_DOMAIN="epoch-supply-shed-viz.app"

# Check if domain mapping already exists
if gcloud run domain-mappings describe ${CUSTOM_DOMAIN} --region=${REGION} &>/dev/null; then
    echo "   Custom domain mapping already exists: https://${CUSTOM_DOMAIN}"
else
    echo "   Creating custom domain mapping..."
    gcloud run domain-mappings create \
        --service=${SERVICE_NAME} \
        --domain=${CUSTOM_DOMAIN} \
        --region=${REGION} || echo "   Domain mapping failed - you may need to verify domain ownership first"
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo "✅ Deployment complete!"
echo "   Default URL: ${SERVICE_URL}"
echo "   Custom URL: https://${CUSTOM_DOMAIN} (if domain mapping succeeded)"
echo "   Login: william@epoch.blue / ssi123"
echo ""
echo "🔧 Service account permissions should already be configured via setup-service-account.sh"
echo "   Using existing service account: 709579113971-compute@developer.gserviceaccount.com"
echo ""
echo "🌐 Domain Setup Notes:"
echo "   - If custom domain failed, you may need to verify domain ownership"
echo "   - Check domain verification in Google Cloud Console"
echo "   - DNS records may need to be updated for the custom domain"
