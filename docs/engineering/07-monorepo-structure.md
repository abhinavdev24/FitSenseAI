# FitSenseAI Monorepo Structure

Last updated: 2026-02-22

## 1. Current Structure (Observed)

- `Data-Pipeline/`: synthetic -> teacher -> distillation pipeline with Airflow DAG and tests
- `database/`: schema artifacts (`tables.sql`, DBML, UML diagram)
- `docs/`: architecture/PRD/system design and supporting runbooks

## 2. New Engineering Docs

This folder:

- `docs/engineering/`: engineering documentation set (12 docs).

## 3. Recommended Additions (When Implementing Backend/Serving)

When you start implementing the online system, add:

- `services/backend/`: backend API service (Cloud Run deploy target)
- `services/inference/`: optional infra and configs for the inference VM (if you want code-managed provisioning)
- `training/`: containerized training entrypoint and configs
- `eval/`: evaluation harness and curated prompt sets

## 4. Documentation Principles

- Keep architecture decisions in `docs/` and engineering specs in `docs/engineering/`.
- Keep runbooks near the relevant component (example: `docs/infra/vllm/`).
- Avoid duplicating the same content in multiple places; reference the source-of-truth doc.

