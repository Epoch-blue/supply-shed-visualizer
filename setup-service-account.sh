#!/bin/bash

# Service Account Setup for Supply Shed Visualizer
# This script configures the existing service account for Cloud Run

set -e

PROJECT_ID="epoch-geospatial-dev"
SERVICE_ACCOUNT_EMAIL="709579113971-compute@developer.gserviceaccount.com"

echo "🔧 Setting up existing service account for Supply Shed Visualizer"
echo "   Project: ${PROJECT_ID}"
echo "   Service Account: ${SERVICE_ACCOUNT_EMAIL}"

# Set the project
gcloud config set project ${PROJECT_ID}

# Grant necessary permissions to existing service account
echo "🔑 Granting permissions to existing service account..."

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

# Create and download service account key (for local development)
echo "🔐 Creating service account key for local development..."
gcloud iam service-accounts keys create ./service-account-key.json \
    --iam-account=${SERVICE_ACCOUNT_EMAIL}

echo "✅ Service account setup complete!"
echo ""
echo "📋 Service Account Details:"
echo "   Email: ${SERVICE_ACCOUNT_EMAIL}"
echo "   Key file: ./service-account-key.json"
echo ""
echo "🔧 For local development, set:"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json"
echo ""
echo "🚀 For Cloud Run deployment, the service will use this service account automatically"
echo "   when you run: ./deploy.sh"
