from dataclasses import dataclass

import numpy as np
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.history_service.models import Turn
from apps.prompt_service.models import BanditUserArmState, Prompt, PromptDecision
from apps.ratings_service.models import TurnEvaluation
from common.errors import PromptDataError, PromptNotFoundError
from common.types import PromptContext, SystemPrompt


DEFAULT_MODEL_VERSION = "pts_normal_v1"
DEFAULT_GAMMA = 0.998
DEFAULT_ALPHA = 1.0
DEFAULT_MU0 = 0.5
DEFAULT_LAMBDA0 = 4.0
DEFAULT_SIGMA_R = 0.2
MAX_PROMPT_STREAK = 3


@dataclass(frozen=True)
class BanditParams:
    gamma: float = DEFAULT_GAMMA
    alpha: float = DEFAULT_ALPHA
    mu0: float = DEFAULT_MU0
    lambda0: float = DEFAULT_LAMBDA0
    sigma_r: float = DEFAULT_SIGMA_R
    model_version: str = DEFAULT_MODEL_VERSION


def _get_setting(name: str, default):
    return getattr(settings, name, default)


def _get_lambda0_setting() -> float:
    # Temporary backward compatibility: BANDIT_LAMBDA0 falls back to BANDIT_LAMBDA.
    if hasattr(settings, 'BANDIT_LAMBDA0'):
        return float(getattr(settings, 'BANDIT_LAMBDA0'))
    return float(_get_setting('BANDIT_LAMBDA', DEFAULT_LAMBDA0))


def _clip_reward(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _posterior_params(state: BanditUserArmState) -> tuple[float, float]:
    lam = max(float(state.lambda0 + state.nu), 1e-12)
    mu = (float(state.lambda0) * float(state.mu0) + float(state.eta)) / lam
    return mu, lam


class PromptService:
    def __init__(self) -> None:
        params = BanditParams(
            gamma=float(_get_setting('BANDIT_GAMMA', DEFAULT_GAMMA)),
            alpha=float(_get_setting('BANDIT_ALPHA', DEFAULT_ALPHA)),
            mu0=float(_get_setting('BANDIT_MU0', DEFAULT_MU0)),
            lambda0=_get_lambda0_setting(),
            sigma_r=float(_get_setting('BANDIT_SIGMA_R', DEFAULT_SIGMA_R)),
            model_version=str(_get_setting('BANDIT_MODEL_VERSION', DEFAULT_MODEL_VERSION)),
        )
        self.params = params

    def select_system_prompt(self, context: PromptContext) -> SystemPrompt:
        if context.conversation_id is None:
            raise PromptDataError("conversation_id required for bandit selection")

        eligible = list(Prompt.objects.filter(is_active=True).order_by('-created_at', '-id'))
        if not eligible:
            raise PromptNotFoundError("No active prompts found.")

        fallback_prompt = eligible[0]
        last_turn = (
            Turn.objects.filter(conversation_id=context.conversation_id)
            .order_by('-turn_index')
            .first()
        )
        turn_number = 0 if last_turn is None else last_turn.turn_index + 1

        best_prompt = None
        best_sample = float('-inf')
        sampled_by_prompt: dict[int, float] = {}
        rng = np.random.default_rng()

        for prompt in eligible:
            state = self._load_or_init_state(context.user_id, prompt)
            mu, lam = _posterior_params(state)
            sampled_theta = float(rng.normal(mu, np.sqrt((state.alpha**2) / lam)))
            sampled_by_prompt[prompt.id] = sampled_theta
            if sampled_theta > best_sample:
                best_sample = sampled_theta
                best_prompt = prompt

        if best_prompt is None:
            best_prompt = fallback_prompt
            best_sample = sampled_by_prompt.get(best_prompt.id, 0.0)

        if self._streak_guardrails(context.conversation_id, best_prompt.id):
            best_prompt = fallback_prompt
            best_sample = sampled_by_prompt.get(best_prompt.id, best_sample)

        self._log_decision(
            learner_id=context.user_id,
            conversation_id=context.conversation_id,
            turn_number=turn_number,
            prompt=best_prompt,
            sampled_theta=best_sample,
        )

        return SystemPrompt(prompt_id=best_prompt.id, text=best_prompt.text)

    def _streak_guardrails(self, conversation_id: str, prompt_id: int) -> bool:
        recent = list(
            Turn.objects.filter(conversation_id=conversation_id)
            .order_by('-turn_index')
            .values_list('prompt_id', flat=True)[:MAX_PROMPT_STREAK]
        )
        if len(recent) < MAX_PROMPT_STREAK:
            return False
        return all(pid == prompt_id for pid in recent)

    def _load_or_init_state(self, learner_id: str, prompt: Prompt) -> BanditUserArmState:
        params = self.params
        state, _ = BanditUserArmState.objects.get_or_create(
            learner_id=learner_id,
            prompt=prompt,
            defaults={
                'mu0': params.mu0,
                'lambda0': params.lambda0,
                'eta': 0.0,
                'nu': 0.0,
                'sigma_r': params.sigma_r,
                'alpha': params.alpha,
                'gamma': params.gamma,
                'effective_n': 0.0,
                'model_version': params.model_version,
            },
        )
        return state

    @transaction.atomic
    def _log_decision(
        self,
        learner_id: str,
        conversation_id: str,
        turn_number: int,
        prompt: Prompt,
        sampled_theta: float,
    ) -> None:
        PromptDecision.objects.create(
            learner_id=learner_id,
            conversation_id=conversation_id,
            prompt=prompt,
            turn_number=turn_number,
            sampled_theta=float(sampled_theta),
            model_version=self.params.model_version,
        )

    def ingest_rewards_and_update(self) -> int:
        updated = 0
        pending = (
            PromptDecision.objects.filter(reward__isnull=True)
            .select_related('prompt', 'turn')
            .order_by('id')
        )
        for decision in pending:
            evaluation_q = self._fetch_reward(decision)
            if evaluation_q is None:
                continue
            applied_reward = self._apply_reward(decision.learner_id, decision.prompt, evaluation_q)
            decision.reward = applied_reward
            decision.reward_computed_at = timezone.now()
            decision.reward_version = 'mean_q_v0'
            decision.save(update_fields=['reward', 'reward_computed_at', 'reward_version'])
            updated += 1
        return updated

    @transaction.atomic
    def apply_reward_for_turn(self, turn: Turn) -> int:
        decisions = PromptDecision.objects.select_for_update().filter(
            turn=turn, reward__isnull=True
        ).select_related('prompt')
        if not decisions.exists():
            return 0

        evaluations = TurnEvaluation.objects.filter(turn=turn)
        scores = [ev.q_total for ev in evaluations]
        if not scores:
            return 0
        reward = float(sum(scores) / len(scores))

        for decision in decisions:
            applied_reward = self._apply_reward(decision.learner_id, decision.prompt, reward)
            decision.reward = applied_reward
            decision.reward_computed_at = timezone.now()
            decision.reward_version = 'mean_q_v0'
            decision.save(update_fields=['reward', 'reward_computed_at', 'reward_version'])
        return decisions.count()

    def _fetch_reward(self, decision: PromptDecision) -> float | None:
        if decision.turn_id is None:
            return None
        evaluations = TurnEvaluation.objects.filter(turn_id=decision.turn_id)
        scores = [ev.q_total for ev in evaluations]
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    def _apply_reward(self, learner_id: str, prompt: Prompt, reward: float) -> float:
        params = self.params
        clipped_reward = _clip_reward(reward)

        with transaction.atomic():
            state, _ = BanditUserArmState.objects.select_for_update().get_or_create(
                learner_id=learner_id,
                prompt=prompt,
                defaults={
                    'mu0': params.mu0,
                    'lambda0': params.lambda0,
                    'eta': 0.0,
                    'nu': 0.0,
                    'sigma_r': params.sigma_r,
                    'alpha': params.alpha,
                    'gamma': params.gamma,
                    'effective_n': 0.0,
                    'model_version': params.model_version,
                },
            )

            sigma_r = float(state.sigma_r) if state.sigma_r > 0.0 else float(params.sigma_r)
            tau_r = 1.0 / (sigma_r**2)
            gamma = float(state.gamma) if state.gamma > 0.0 else float(params.gamma)

            state.eta = gamma * float(state.eta) + tau_r * clipped_reward
            state.nu = gamma * float(state.nu) + tau_r
            state.effective_n = gamma * float(state.effective_n) + 1.0
            state.save(update_fields=['eta', 'nu', 'effective_n', 'updated_at'])

        return clipped_reward
