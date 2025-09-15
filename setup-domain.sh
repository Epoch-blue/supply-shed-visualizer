#!/bin/bash

# Custom Domain Setup for Supply Shed Visualizer
# This script helps set up a custom domain for the Cloud Run service

set -e

PROJECT_ID="epoch-geospatial-dev"
SERVICE_NAME="supply-shed-visualizer"
REGION="us-central1"
CUSTOM_DOMAIN="epoch-supply-shed-viz.app"

echo "🌐 Setting up custom domain for Supply Shed Visualizer"
echo "   Project: ${PROJECT_ID}"
echo "   Service: ${SERVICE_NAME}"
echo "   Domain: ${CUSTOM_DOMAIN}"
echo "   Region: ${REGION}"

# Set the project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com

# Create domain mapping
echo "🌐 Creating domain mapping..."
gcloud run domain-mappings create \
    --service=${SERVICE_NAME} \
    --domain=${CUSTOM_DOMAIN} \
    --region=${REGION}

# Get the DNS records that need to be added
echo "📋 DNS Configuration Required:"
echo ""
echo "You need to add the following DNS records to your domain (${CUSTOM_DOMAIN}):"
echo ""

# Get the domain mapping details
DOMAIN_MAPPING=$(gcloud run domain-mappings describe ${CUSTOM_DOMAIN} --region=${REGION} --format="value(status.resourceRecords[].name,status.resourceRecords[].type,status.resourceRecords[].rrdata)" | tr '\t' ' ')

echo "DNS Records to add:"
echo "==================="
echo "${DOMAIN_MAPPING}"
echo ""

echo "📝 Next Steps:"
echo "1. Add the DNS records above to your domain registrar"
echo "2. Wait for DNS propagation (can take up to 24 hours)"
echo "3. Verify domain ownership in Google Cloud Console if needed"
echo "4. Test the custom domain: https://${CUSTOM_DOMAIN}"
echo ""
echo "🔍 To check domain mapping status:"
echo "   gcloud run domain-mappings describe ${CUSTOM_DOMAIN} --region=${REGION}"
echo ""
echo "✅ Domain mapping created successfully!"
echo "   Custom URL: https://${CUSTOM_DOMAIN}"
