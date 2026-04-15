# Vertex AI deployment guide for FitSenseAI

This guide is for the recommended production path:

- keep a **stable Vertex endpoint**
- update the deployed model version behind that endpoint
- keep the backend code unchanged except for the endpoint environment variables

## 1. What you deploy

Deploy a **custom inference container** to Vertex AI. The container lives in:

- `backend/cloud/vertex_inference/`

It exposes a `/predict` route that Vertex forwards prediction requests to.

## 2. Required tools

- Google Cloud CLI (`gcloud`)
- Docker
- Access to a Google Cloud project with Vertex AI and Artifact Registry enabled
- Permission to push images and deploy Vertex models/endpoints

## 3. Environment variables

Set these in your shell before using the commands below:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export REPOSITORY="fitsenseai-models"
export IMAGE_NAME="fitsense-vertex-inference"
export IMAGE_TAG="v1"
export SERVICE_ACCOUNT="vertex-inference@${PROJECT_ID}.iam.gserviceaccount.com"
export MODEL_DISPLAY_NAME="fitsense-student"
export ENDPOINT_DISPLAY_NAME="fitsense-student-endpoint"
export ARTIFACT_URI="gs://fitsenseai-model-registry/models/fitsense-qwen3-4b/20260401Z/"
```

`ARTIFACT_URI` should point to the GCS folder that contains the adapter package or files that the inference container can download at startup.

## 4. Enable APIs

```bash
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

## 5. Create an Artifact Registry Docker repo

```bash
gcloud artifacts repositories create "$REPOSITORY"   --repository-format=docker   --location="$REGION"   --description="FitSenseAI Vertex inference images"
```

If it already exists, you can skip this.

## 6. Build and push the inference image

From the repo root:

```bash
cd backend/cloud/vertex_inference

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

docker build -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" .
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
```

## 7. Upload the model to Vertex Model Registry

```bash
gcloud ai models upload   --region="$REGION"   --display-name="$MODEL_DISPLAY_NAME"   --container-image-uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"   --artifact-uri="$ARTIFACT_URI"   --container-predict-route="/predict"   --container-health-route="/health"
```

## 8. Create an endpoint

```bash
gcloud ai endpoints create   --region="$REGION"   --display-name="$ENDPOINT_DISPLAY_NAME"
```

Copy the endpoint ID from the command output.

## 9. Deploy the model to the endpoint

Get the model ID from the previous upload command or from the Vertex console, then run:

```bash
gcloud ai endpoints deploy-model ENDPOINT_ID   --region="$REGION"   --model=MODEL_ID   --display-name="fitsense-student-v1"   --machine-type="g2-standard-8"   --accelerator=type=nvidia-l4,count=1   --min-replica-count=1   --max-replica-count=1   --traffic-split=0=100   --service-account="$SERVICE_ACCOUNT"
```

Adjust machine and accelerator to what your quota supports.

## 10. Point the backend to Vertex

Set these in the backend environment:

```bash
export FITSENSE_VERTEX_PROJECT="$PROJECT_ID"
export FITSENSE_VERTEX_REGION="$REGION"
export FITSENSE_VERTEX_ENDPOINT_ID="ENDPOINT_ID"
```

Then start the backend.

## 11. How model updates work

When you train a newer student model:

1. push the new model artifact to GCS
2. upload a **new Vertex model version** (or a new model resource)
3. deploy the new version to the **same endpoint**
4. keep the backend pointing at the same `ENDPOINT_ID`

That way, you do not need backend code changes for each model update.
