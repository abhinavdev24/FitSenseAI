# FitSense AI — Landing Page

> Next.js 14 marketing site for FitSense AI, an AI-powered fitness coaching app built on a fine-tuned student model trained via teacher-student distillation.

[![Live Site](https://img.shields.io/badge/Live%20Site-fitsenseai.abhinavdev24.com-2563EB?style=for-the-badge&logo=googlechrome&logoColor=white)](https://fitsenseai.abhinavdev24.com/)

|                Preview                |                     Preview                     |
| :-----------------------------------: | :---------------------------------------------: |
|     ![Hero](screenshots/hero.png)     |      ![Features](screenshots/features.png)      |
| ![Personas](screenshots/personas.png) | ![Meet the Team](screenshots/meet-the-team.png) |

---

## Overview

FitSense AI turns your goals, constraints, and weekly performance into a structured, adapting workout plan. The landing page communicates what the app does, how the AI works, and who it's for — and serves the Android APK for direct download.

---

## Architecture

### System Architecture

```mermaid
graph TD
    subgraph Landing ["Landing Page (Next.js 14)"]
        direction TB
        N[Navbar] --> H[Hero]
        H --> AS[App Screenshots]
        AS --> VS[Video Section]
        VS --> F[Features]
        F --> AB[About]
        AB --> P[Personas]
        P --> MT[Meet the Team]
        MT --> CTA[CTA Banner]
        CTA --> FT[Footer]
    end

    subgraph Mobile ["Mobile App (Flutter)"]
        Dashboard
        YourPlan[Your Plan]
        LogSession[Log Session]
        DailyCheckin[Daily Check-in]
        AICoach[AI Coach Chat]
    end

    subgraph Backend ["Online Plane (Cloud Run)"]
        API[Backend API]
        DB[(Cloud SQL)]
        LLM[vLLM · Student Model]
    end

    subgraph MLPipeline ["Offline Plane (Data Pipeline)"]
        DP[Phases 1–6\nSynthetic Data]
        Teacher[Teacher LLM\nQwen3:32b · Groq]
        Distill[Distillation\n+ LoRA Fine-tune]
        StudentModel[Student Model\nQwen3:8b]
    end

    Landing -->|Download APK| Mobile
    Mobile --> API
    API --> DB
    API --> LLM
    Teacher --> Distill
    DP --> Teacher
    Distill --> StudentModel
    StudentModel --> LLM
```

## App Screens

FitSense AI is a five-tab Flutter application:

| Tab                | Purpose                                                           |
| ------------------ | ----------------------------------------------------------------- |
| **Dashboard**      | Goal summary, avg sleep/calories/weight, recent workouts          |
| **Your Plan**      | AI-generated weekly structure with sets · reps · RIR per exercise |
| **Log Session**    | Per-set logging of reps, weight, RIR — feeds next-week adaptation |
| **Daily Check-in** | Sleep hours, calorie intake, body weight signals                  |
| **AI Coach**       | Conversational coach scoped to fitness, safety-aware              |

---

## Features

| Feature                   | Description                                                                       |
| ------------------------- | --------------------------------------------------------------------------------- |
| **Personalized Plans**    | Structured weekly program tailored to your goal, equipment, and constraints       |
| **AI Adaptation**         | Plan evolves each week based on logged performance — volume, intensity, exercises |
| **Workout Logging**       | Sets · reps · weight · RIR with a clean, distraction-free interface               |
| **Daily Check-ins**       | Sleep, calories, and body weight feed the adaptation engine                       |
| **Safety-Aware Coaching** | Injury flags and medical constraints respected at every step; escalates on risk   |
| **AI Coach Chat**         | Ask anything about training — technique, scheduling, substitutions                |

---

## Who It's For

```mermaid
graph LR
    subgraph Beginner ["🌱 Beginner Lifter"]
        B1[Clear structure]
        B2[Guided progressions]
        B3[No guesswork]
    end
    subgraph Intermediate ["💪 Intermediate Lifter"]
        I1[Smart progression]
        I2[Performance-driven]
        I3[Weekly adaptation]
    end
    subgraph Constrained ["🩺 Constraint-Heavy User"]
        C1[Injury-aware]
        C2[Equipment flexibility]
        C3[Safety-first]
    end

    FitSenseAI[FitSense AI] --> Beginner
    FitSenseAI --> Intermediate
    FitSenseAI --> Constrained
```

## Getting Started (Local Dev)

```bash
cd landing
npm install
npm run dev
# → http://localhost:3000
```

### Build & Export

```bash
npm run build
```

### Project Structure

```
landing/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Home (all sections)
│   │   ├── about/page.tsx    # Meet the Team route
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── components/
│       ├── Navbar.tsx
│       ├── Hero.tsx
│       ├── AppScreenshots.tsx
│       ├── VideoSection.tsx
│       ├── Features.tsx
│       ├── About.tsx
│       ├── Personas.tsx
│       ├── MeetTeam.tsx
│       ├── CTABanner.tsx
│       └── Footer.tsx
├── public/
│   ├── logo.svg
│   └── fitsense.apk          # Android APK served for download
├── screenshots/
│   ├── hero.png
│   ├── features.png
│   ├── personas.png
│   └── meet-the-team.png
├── next.config.mjs
├── tailwind.config.ts
└── package.json
```
