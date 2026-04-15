# Cloud Run deployment guide for FitSenseAI student model

This is the simplest path if you want to serve the trained student model with **Cloud Run instead of Vertex AI**.

## Architecture

- **Cloud Run inference service** hosts the model and exposes:
  - `GET /health`
  - `POST /predict`
- **FitSense backend** calls that Cloud Run service using `FITSENSE_CLOUDRUN_URL`
- **Flutter/mobile app** keeps talking to the backend exactly like before

That means the app does **not** call Cloud Run directly.

---

## 1. What code is already ready

Use this service directory:

- `backend/cloud/vertex_inference/`

Even though the folder says `vertex_inference`, it is just a FastAPI model server and works for Cloud Run too.

---

## 2. Prerequisites

Install and set up:

- Google Cloud SDK (`gcloud`)
- Docker
- A Google Cloud project with billing enabled

Enable the required APIs:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com
```

---

## 3. Set shell variables

Run these in your terminal:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export REPOSITORY="fitsenseai-services"
export IMAGE_NAME="fitsense-student-inference"
export IMAGE_TAG="v1"
export SERVICE_NAME="fitsense-student-inference"
export MODEL_GCS_URI="gs://YOUR_BUCKET/qwen3-4b-v2.zip"
```

`MODEL_GCS_URI` can point to:
- a `.zip` uploaded to GCS, or
- a folder in GCS containing adapter files

---

## 4. Upload your model to Google Cloud Storage

If your model zip is local, upload it first:

```bash
gcloud storage cp qwen3-4b-v2.zip "$MODEL_GCS_URI"
```

Example:

```bash
gcloud storage cp qwen3-4b-v2.zip gs://fitsenseai-model-registry/qwen3-4b-v2.zip
```

---

## 5. Create Artifact Registry repo

```bash
gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION" \
  --description="FitSenseAI Cloud Run images"
```

If it already exists, skip this.

---

## 6. Build and push the inference container

From the repo root:

```bash
cd backend/cloud/vertex_inference

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

docker build -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" .
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
```

---

## 7. Deploy the model server to Cloud Run

### CPU-only version

Good for first test, but slower for a 4B model.

```bash
gcloud run deploy "$SERVICE_NAME" \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 900 \
  --concurrency 1 \
  --set-env-vars "FITSENSE_ADAPTER_GCS_URI=${MODEL_GCS_URI}"
```

### More secure version

If you do **not** want public access, remove `--allow-unauthenticated` and let only your backend service account invoke it.
The backend code in this repo now supports authenticated Cloud Run calls using an identity token.

---

## 8. Get the Cloud Run URL

After deployment:

```bash
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format='value(status.url)'
```

You will get something like:

```bash
https://fitsense-student-inference-xxxxx-uc.a.run.app
```

---

## 9. Point the backend to Cloud Run

Set this in your backend environment:

```bash
export FITSENSE_CLOUDRUN_URL="https://fitsense-student-inference-xxxxx-uc.a.run.app"
```

Then start the backend.

The backend will call:

```text
${FITSENSE_CLOUDRUN_URL}/predict
```

---

## 10. Test the Cloud Run service directly

### Health

```bash
curl "${FITSENSE_CLOUDRUN_URL}/health"
```

### Predict

```bash
curl -X POST "${FITSENSE_CLOUDRUN_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "task": "coach_text",
        "system_prompt": "You are a fitness coach.",
        "user_message": "I feel sore after leg day. What should I do?",
        "max_new_tokens": 120
      }
    ]
  }'
```

---

## 11. Run your backend locally against Cloud Run

From `backend/`:

```bash
export FITSENSE_CLOUDRUN_URL="https://fitsense-student-inference-xxxxx-uc.a.run.app"
uvicorn app.main:app --reload
```

Then check:

```bash
curl http://127.0.0.1:8000/
```

And after login, inspect runtime:

```bash
curl http://127.0.0.1:8000/model/runtime
```

You should see provider information showing Cloud Run.

---

## 12. Model updates later

When your model changes:

1. upload the new zip/folder to GCS
2. redeploy Cloud Run with the new `FITSENSE_ADAPTER_GCS_URI`, or keep the same URI and overwrite the artifact
3. backend code stays the same as long as `FITSENSE_CLOUDRUN_URL` stays the same

---

## 13. Important note about performance

A 4B model on Cloud Run CPU can be very slow and may fail if memory is too low.

For a quick project demo:
- Cloud Run is simpler than Vertex
- start with CPU just to validate the full pipeline
- if latency is too high, move to a GPU-backed serving path later

---

## 14. Backend change already made

`backend/app/llm_runtime.py` now supports:

- `FITSENSE_CLOUDRUN_URL` for Cloud Run inference
- authenticated Cloud Run invocation using Google identity tokens
- fallback to local adapter if cloud inference fails

