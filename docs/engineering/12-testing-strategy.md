# FitSenseAI Testing Strategy

Last updated: 2026-02-22

## 1. Goals

- Prevent safety regressions.
- Ensure reproducibility and data quality in the offline pipeline.
- Verify online behavior under overload and streaming constraints.

## 2. Existing Tests (Pipeline)

Current tests live in `Data-Pipeline/tests/` and cover:

- reproducibility
- synthetic data generation bounds and schema checks
- query generation coverage and metadata integrity
- teacher call output contracts
- distillation split integrity
- phase 6 report generation

## 3. Backend Tests (To Add)

- Unit tests:
  - request validation
  - auth and per-user isolation
  - DB persistence logic for workouts and logs
- Integration tests:
  - plan generation path with mocked inference responses
  - streaming proxy behavior
  - overload behavior (queue or 429)

## 4. Inference Tests (To Add)

- Health check tests for vLLM endpoint (private network).
- Contract tests for `/v1/chat/completions` streaming.
- Load tests:
  - verify stable behavior at target peak concurrency with queueing/backpressure.

## 5. Safety and Quality Evaluation (To Add)

- Curated prompt sets:
  - injury/pain prompts
  - contraindicated request prompts
  - plan modification prompts that test constraints
- Rule-based gates:
  - no unsafe instruction patterns
  - minimum content requirements

## 6. CI Gates (Suggested)

- Pipeline: run `pytest Data-Pipeline/tests`.
- Backend: run unit and integration tests.
- Safety: run curated prompt eval and fail CI if safety gate fails.

