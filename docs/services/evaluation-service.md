# Evaluation Service

File: `src/apps/evaluation_service/service.py`

## Purpose
Separates Q-score evaluation logic from feedback ingestion.

## Responsibilities
- Resolve evaluator config (`Evaluator`).
- Normalize ratings and compute Q-components.
- Persist/update `TurnEvaluation`.

## Scoring
Default weighted score (`qscore_v2`):
- progress: 0.4737
- confusion_reduction: 0.2632
- clarity: 0.1579
- engagement: 0.1053

Component details:
- `q_progress`: delta score from perceived progress vs previous rated turn.
- `q_confusion_reduction`: delta score from clarity/understanding vs previous rated turn.
- `q_clarity`: normalized current clarity/understanding rating.
- `q_engagement`: normalized current engagement-fit rating.

Delta score mapping:
- `delta = current_norm - previous_norm`
- `score = clip(0.5 + 0.5 * delta, 0, 1)`
- first rated turn fallback: `0.5` (neutral)

## Main Methods
- `evaluate_turn(turn)`
- `evaluate_turn_with_reward_payload(turn)`
