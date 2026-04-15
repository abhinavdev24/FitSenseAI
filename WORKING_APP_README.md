# FitSenseAI Working App + Backend

This package keeps the existing `Data-Pipeline/`, docs, and database design, and adds a runnable local MVP for the **online** part of the architecture:

- `mobile_app/` → Flutter client for Android / iOS / Web / desktop
- `backend/` → FastAPI backend with local SQLite storage

## What was completed

### Backend
- local auth (`/auth/signup`, `/auth/login`)
- onboarding/profile persistence (`/profile/onboarding`)
- plan generation and modification (`/plans`, `/plans/current`, `/plans/{plan_id}:modify`)
- workout logging (`/workouts`, `/workouts/{id}/exercises`, `/workouts/{id}/sets`)
- daily logs (`/daily/sleep`, `/daily/calories`, `/daily/weight`)
- dashboard aggregation (`/dashboard`)
- coaching endpoint (`/coach`) + SSE stream endpoint (`/coach/stream`)
- next-week adaptation endpoint (`/adaptation:next_week`)

### Flutter app
- login / signup
- complete onboarding profile flow
- dashboard
- current plan view + modify-plan box
- workout session logger
- daily check-in tab
- coach chat tab
- settings + adaptation screen

## Folder map

- `backend/`
  - `app/main.py` → FastAPI routes
  - `app/models.py` → database models
  - `app/services.py` → plan generation, coaching, adaptation logic
  - `data/fitsense.db` → created automatically at runtime
- `mobile_app/`
  - `lib/screens/auth/` → auth screens
  - `lib/screens/onboarding/` → profile setup
  - `lib/screens/home/` → dashboard, plan, workout, check-in, coach, settings
  - `lib/services/` → API and session helpers
  - `lib/legacy/` → your original partial Flutter screens preserved for reference

## Run commands

### 1) Start backend

```bash
cd FitSenseAI-main/backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health/root: `http://127.0.0.1:8000/`

### 2) Start Flutter app

```bash
cd FitSenseAI-main/mobile_app
flutter pub get
flutter run
```

### 3) Web run (optional)

```bash
cd FitSenseAI-main/mobile_app
flutter run -d chrome
```

## Local backend base URL used by the Flutter app

- Android emulator: `http://10.0.2.2:8000`
- iOS simulator / macOS / Linux / Windows desktop: `http://127.0.0.1:8000`
- Web: `http://localhost:8000`

## Notes

- This local MVP intentionally **does not** implement the cloud deployment pieces or real model training.
- The coaching and plan-modification logic is implemented with a local rule engine so the app works end-to-end without a deployed model server.
- The `Data-Pipeline/` folder from your original repository is preserved unchanged for the offline pipeline side.
