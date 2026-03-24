# Ratings Service

File: `src/apps/ratings_service/service.py`

## Purpose
Ingests feedback for a turn, triggers evaluation, and applies prompt reward updates.

## Responsibilities
- Persist `TurnFeedback`.
- Trigger embedding for feedback text (or enqueue retry).
- Compute turn evaluation through `EvaluationService`.
- Apply bandit reward updates via `PromptService`.
- Emit feedback/evaluation/bandit observability events.

## Main Entry
`record_feedback_and_evaluate(turn_id, user_id, ratings..., free_text, trace_id)`

Ratings payload (`qscore_v2` hard cut):
- `rating_perceived_progress`
- `rating_clarity_understanding`
- `rating_engagement_fit`

## Outputs
- `(TurnFeedback, TurnEvaluation | None)`
