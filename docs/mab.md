## 0) Design choices (lock these in)

### Bandit scope

* **Per-user Thompson Sampling**.
* For learner `L`, keep separate bandit state for each prompt arm `a`.
* No explicit context vector (`x`) is used.

### Arms

* `arms(L) = prompt variants that learner L is eligible to receive`.
* In practice, this is usually all active prompts, with learner-specific state created lazily.

### Reward

* `r ∈ [0,1]` = Q-score for the chosen prompt on that turn.
* If multiple evaluations exist for the same turn:
  * `r = mean(q_scores)` (v2 runtime uses reward version `q_v2`).

### Core behavior

* Learn which prompts work best for each learner.
* Reward feedback drives personalization.
* No context features required.

---

## 1) Probabilistic model (distributions)

We model each learner-arm pair `(L, a)` independently.

### Parameter we want to learn

* `theta_{L,a}` = latent mean reward for learner `L` on prompt `a`.

### Likelihood (observation model)

For each observed reward `r_t` from learner `L` using arm `a`:

* `r_t | theta_{L,a} ~ Normal(theta_{L,a}, sigma_r^2)`

Notes:

* Q-scores are bounded in `[0,1]`; this Gaussian likelihood is a practical approximation for continuous rewards.
* Keep `r_t` clipped to `[0,1]` before updates.

### Prior

For each learner-arm pair:

* `theta_{L,a} ~ Normal(mu0_a, 1/lambda0_a)`

Where:

* `mu0_a` is prior mean for prompt `a` (often `0.5` or a global arm mean).
* `lambda0_a` is prior precision (inverse variance), controlling how conservative cold-start behavior is.

### Posterior

Using Normal prior + Normal likelihood gives a Normal posterior:

* `theta_{L,a} | data ~ Normal(mu_{L,a}, 1/lambda_{L,a})`

with discounted sufficient statistics:

* `eta_{L,a}` = discounted precision-weighted reward sum
* `nu_{L,a}` = discounted precision sum
* `tau_r = 1 / sigma_r^2` (observation precision)

Posterior parameters:

* `lambda_{L,a} = lambda0_a + nu_{L,a}`
* `mu_{L,a} = (lambda0_a * mu0_a + eta_{L,a}) / lambda_{L,a}`

### Thompson Sampling step

At decision time, for each arm `a`:

1. Sample `tilde_theta_{L,a} ~ Normal(mu_{L,a}, alpha^2 / lambda_{L,a})`
2. Pick arm with highest sample.

`alpha` is an exploration scale (typically `1.0` to start).

---

## 2) Discounting (short-term stationarity)

Use discount factor `gamma` to prioritize recent rewards for each learner.

For each resolved reward `r` on `(L, a)`:

* `eta_{L,a} <- gamma * eta_{L,a} + tau_r * r`
* `nu_{L,a} <- gamma * nu_{L,a} + tau_r`

This yields recency-weighted personalization without explicit features.

Recommended v0:

* `gamma` in `[0.995, 0.999]` if updates are frequent.
* Move closer to `1.0` when updates are sparse.

---

## 3) Data model additions (minimum viable)

### A) Decision logging

Create/adjust table: `PromptDecision`

* `id`
* `learner_id`
* `conversation_id`
* `turn_id`
* `prompt_id` (chosen arm)
* `sampled_theta` (sample used for selection)
* `chosen_at`
* `reward` (nullable, filled after eval join)
* `reward_computed_at` (nullable)
* `reward_version` (e.g. `"q_v2"`)
* `model_version` (e.g. `"pts_normal_v1"`)

Drop contextual fields from this design (`context_vector`, `context_version`, etc.).

### B) Per learner-arm posterior state

Create table: `BanditUserArmState`

* `learner_id`
* `prompt_id`
* `mu0` (prior mean for this arm)
* `lambda0` (prior precision for this arm)
* `eta` (discounted weighted reward sum)
* `nu` (discounted weighted precision sum)
* `sigma_r` (observation std dev)
* `alpha` (TS sample scale)
* `gamma` (discount factor)
* `updated_at`
* `effective_n` (optional monitoring proxy)

Primary key:

* `(learner_id, prompt_id)`

Initialization:

* `eta = 0`
* `nu = 0`
* posterior starts at the prior.

---

## 4) Online selection algorithm (Per-user TS)

When prompt service picks a prompt for learner `L`:

1. Load eligible prompts for `L`.
2. For each prompt `a`, load `BanditUserArmState(L, a)` or initialize from arm prior.
3. Compute posterior params:
   * `lambda = lambda0 + nu`
   * `mu = (lambda0 * mu0 + eta) / lambda`
4. Sample:
   * `theta_sample ~ Normal(mu, alpha^2 / lambda)`
5. Choose arm with max sampled value.
6. Write `PromptDecision`.

Guardrails:

* Keep a safe fallback prompt eligible.
* Optional repetition cap (for example, no more than 3 repeats in a row).
* Optional new-arm traffic cap.

---

## 5) Reward ingestion + posterior update

Join path:

* `PromptDecision.turn_id -> Turn.id`
* `Turn.id -> Evaluation.turn_id`

Resolved decision:

* has at least one `q_score`.

Aggregate:

* `r = mean(q_scores)` and clip to `[0,1]`.

Update learner-arm state `(L, a)`:

* `tau_r = 1 / sigma_r^2`
* `eta <- gamma * eta + tau_r * r`
* `nu <- gamma * nu + tau_r`

Then recompute posterior on read:

* `lambda = lambda0 + nu`
* `mu = (lambda0 * mu0 + eta) / lambda`

This update is idempotent when each decision is resolved once.

---

## 6) Why this still works (no explicit context)

* The learner identity is the effective context.
* Short-term learner cognition is approximately stationary.
* Reward feedback continuously adapts prompt preferences.

Equivalent interpretation:

* Personalized reinforcement learning without feature engineering.

---

## 7) Hyperparameters (what they mean and what they do)

These control how quickly the model adapts, how much it explores, and how stable/cautious it is.

### `mu0` (prior mean)

* Default: `mu0 = 0.5`
* Meaning: baseline expected reward for a new learner-arm pair before any data.
* Effect:
  * Higher `mu0` starts new arms more optimistic.
  * Lower `mu0` starts new arms more pessimistic.
* Use:
  * Set near the long-run average Q-score if known.
  * Keep at `0.5` if you want neutral cold-start behavior.

### `lambda0` (prior precision)

* Default: `lambda0 = 4.0` (prior variance `= 1/lambda0 = 0.25`)
* Meaning: strength of the prior relative to incoming rewards.
* Effect:
  * Higher `lambda0` = stronger prior, slower movement from early rewards.
  * Lower `lambda0` = weaker prior, faster adaptation (more sensitivity/noise).
* Use:
  * Increase when early reward noise is high.
  * Decrease when you want faster personalization for new learners.

### `sigma_r` (likelihood noise scale)

* Default: `sigma_r = 0.2` so `tau_r = 1/sigma_r^2 = 25`
* Meaning: assumed noise in observed Q-scores around true `theta`.
* Effect:
  * Smaller `sigma_r` (larger `tau_r`) makes each reward update stronger.
  * Larger `sigma_r` (smaller `tau_r`) makes updates more conservative.
* Use:
  * If reward labels are stable and trustworthy, lower `sigma_r`.
  * If rewards are noisy/inconsistent, raise `sigma_r`.

### `alpha` (Thompson exploration scale)

* Default: `alpha = 1.0`
* Meaning: multiplies posterior sampling variance during action selection.
* Effect:
  * Higher `alpha` increases exploration (more random arm switching).
  * Lower `alpha` increases exploitation (more greedy behavior).
* Use:
  * Raise if the policy converges too early on suboptimal prompts.
  * Lower if behavior is too volatile for production UX.

### `gamma` (discount factor)

* Default: `gamma = 0.998`
* Meaning: recency weighting in updates (`eta <- gamma*eta + ...`, `nu <- gamma*nu + ...`).
* Effect:
  * Higher `gamma` (closer to `1`) keeps longer memory, slower adaptation.
  * Lower `gamma` forgets older data faster, adapts more quickly.
* Use:
  * Lower `gamma` when learner behavior drifts quickly.
  * Raise `gamma` when behavior is stable and data is sparse.

### Practical tuning order

1. Tune `sigma_r` and `lambda0` first (update stability).
2. Tune `gamma` second (adaptation speed to drift).
3. Tune `alpha` last (exploration vs exploitation at serving time).
4. Keep `mu0` near global average Q-score unless cold-start behavior is clearly biased.

Validate all choices with offline replay before changing production defaults.
