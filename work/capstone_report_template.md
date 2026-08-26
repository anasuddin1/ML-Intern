# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Anas Uddin
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/anasuddin1/ML-Intern/
- **Date:** August 26, 2026


## 1. Problem framing

**Unit of analysis:** one client-content observation, aggregated across a full calendar month (client_hash_id + content_hash_id pair).

**Output:** a ranked review queue. Each observation receives a model probability, an evidence-aware priority score, a reason code, and an archetype label.

**Action a human takes:** a content or SEO reviewer works down the queue and investigates the highest-priority observations first — checking title/snippet alignment, content experience, or refresh opportunity depending on the archetype. The workflow explicitly requires human review; no action is applied automatically.

**Cost of a wrong call:** a false positive wastes reviewer time investigating a page that wasn't actually declining. A false negative means a genuinely declining page is missed and continues losing clicks unnoticed. Given the review cost is low (time) relative to the cost of missing real decline, the workflow favors recall alongside evidence-aware filtering to avoid drowning reviewers in low-evidence noise.

**Why ML helps here:** with 331,437+ aggregated content observations per month, no team can manually review every page every month. A transparent, probabilistic scoring layer lets reviewers prioritize limited attention on the observations most associated with next-month decline, rather than reviewing pages at random or by gut feel.

## 2. Data safety

**Data used:** `FlyRank/internship-warehouse`, table `fact_content_daily_performance`, two monthly partitions only:
- March 2026 (`month=2026-03/data_0.parquet`) — 9,841,378 daily rows, feature window
- April 2026 (`month=2026-04/data_0.parquet`) — 10,424,730 daily rows, outcome window

Columns used for March features: `gsc_impressions`, `gsc_clicks`, `gsc_avg_position`, `ga4_sessions`, `ga4_engaged_sessions`, `sessions_organic`, `sessions_ai`, `scroll_events`, plus derived `march_ctr` and `march_engagement_rate`.

Columns used for April: `gsc_impressions`, `gsc_clicks`, `ga4_sessions`, `ga4_engaged_sessions` — used only to construct the outcome label, never as model features.

**Deliberately excluded:**
- `client_hash_id`, `content_hash_id` as numerical model features — used only as the grouping variable for `GroupShuffleSplit`, never passed to the model.
- All April fields as features (outcome-window data cannot inform the feature window).
- `future_decline` (the target itself) is excluded from the feature matrix `X`.
- `baseline_action_score`, `reason_code`, `action`, `refresh_tier`, `health_score`, `priority_score` — all downstream/derived product-decision fields, none used as predictive features.

**Leakage risks considered:** the target `future_decline` is defined purely from April outcome fields compared against March clicks — it cannot leak into March-only features by construction. The grouped split ensures no client appears in both train and test, which prevents the model from learning client-specific quirks and then being evaluated on the same clients (a common false-positive source of inflated validation metrics).

**Confirmation:** no client names, domains, URLs, private search queries, or credentials appear anywhere in `work/`. All identifiers are pseudonymous hashes used only for grouping.

## 3. Baseline

**Baseline used:** a majority-class probability baseline. It predicts the training-set positive rate (0.6530) for every test observation, regardless of that observation's features.

**Why this is a fair comparison:** it is evaluated on the exact same client-grouped test split as the model (5,062 observations, 9 held-out clients, zero client overlap with training). It represents the naive floor — the performance achievable with no feature information at all, only knowledge of the overall class balance.

**Baseline numbers on the same split:**

| Metric | Baseline |
|---|---|
| ROC-AUC | 0.5000 |
| Average precision | 0.6831 |
| Brier score | 0.2174 |
| Accuracy @ 0.50 | 0.6831 |

## 4. Model / analysis

**Method:** Logistic Regression, chosen because it is transparent (coefficients can be inspected directly), produces calibrated probabilities suitable for ranking, is relatively simple, and does not reward complexity for its own sake — appropriate as a first supervised model for this lane.

**Pipeline:** `SimpleImputer(strategy="median")` → `StandardScaler()` → `LogisticRegression(max_iter=1000, random_state=42)`.

**Feature list (10 features, March-only):**
`march_impressions`, `march_clicks`, `march_avg_position`, `march_sessions`, `march_engaged_sessions`, `march_organic_sessions`, `march_ai_sessions`, `march_scroll_events`, `march_ctr`, `march_engagement_rate`.

**Left out on purpose:** client/content identifiers, all April fields, any downstream product-decision flags (reason codes, action labels, health/priority scores) — these either leak the future, leak identity, or leak the label itself.

**Target definition, one sentence:** `future_decline = 1` when a client-content observation had March clicks greater than zero and April clicks lower than March clicks; observations with zero March clicks are excluded because the outcome requires measurable prior demand.

## 5. Evaluation

**Split:** `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`, grouped by `client_hash_id`. A client-grouped split was used rather than a random row split because the unit of business risk is the client — a random split would let the model see some of a client's pages in training and others in test, silently leaking client-specific baseline behavior and overstating real-world performance.

**Split numbers:** train 63,775 rows / 35 clients; test 5,062 rows / 9 clients; 0 client overlap between train and test.

**Model vs baseline, same split:**

| Method | ROC-AUC | Avg Precision | Brier Score | Accuracy @0.50 |
|---|---|---|---|---|
| Majority baseline | 0.5000 | 0.6831 | 0.2174 | 0.6831 |
| Logistic Regression | **0.6132** | **0.7813** | **0.2110** | 0.6839 |

**Base rate:** the test-set positive (decline) rate is 68.31% — this is the number any accuracy or precision figure must be read against. The model's ROC-AUC and average precision gains over the baseline are the honest discrimination signal, since accuracy alone is barely distinguishable from the base rate.

**Error analysis:** at the 0.50 threshold, the confusion matrix showed 1,599 false positives and only 1 false negative. In plain terms: the model predicted almost every observed decline correctly (very high recall, ~0.9997) but at the cost of many false alarms — pages flagged as likely to decline that did not. This tradeoff is acceptable for a *prioritization* tool (missing a real decline is costlier than one extra review), but it means the raw probability output should not be read as a precise decline forecast — it is better used for ranking than for binary classification at a fixed threshold.

## 6. Interpretation

Standardized coefficients (directional associations, not causal effects), strongest to weakest by magnitude:

| Feature | Coefficient | Direction |
|---|---|---|
| march_ctr | +0.7566 | higher CTR associated with more decline risk |
| march_sessions | −0.1242 | more sessions associated with less decline risk |
| march_scroll_events | +0.0938 | more scroll events associated with more decline risk |
| march_organic_sessions | −0.0603 | more organic sessions associated with less decline risk |
| march_impressions | +0.0580 | more impressions associated with more decline risk |
| march_engaged_sessions | +0.0553 | more engaged sessions associated with more decline risk |
| march_engagement_rate | −0.0426 | higher engagement rate associated with less decline risk |
| march_ai_sessions | −0.0223 | more AI-referral sessions associated with less decline risk |
| march_clicks | −0.0175 | more clicks associated with less decline risk |
| march_avg_position | +0.0094 | weakest feature, near-negligible association |

**In plain words:** the strongest single signal is CTR — pages with an already-high March CTR were, somewhat counterintuitively, more associated with a subsequent click decline in this dataset. This could reflect regression-to-the-mean (pages with an unusually strong CTR in March have more room to fall) rather than any causal mechanism. Session volume and organic session share point the other direction — more overall traffic and engagement associate with lower decline risk, which is more intuitive (established, well-trafficked pages are more stable).

**Surprise / negative result:** `march_avg_position`, often assumed to be a dominant SEO signal, had the weakest association of all ten features in this model. That is a legitimate negative result worth reporting rather than omitting.

**Retrospective visibility pattern (descriptive, not causal):** observed April decline rate was 83.2% for low-visibility observations (0–99 March impressions, n=5,175), 69.0% for medium visibility (100–499, n=12,534), and 62.9% for high visibility (≥500, n=51,128). Lower-visibility pages showed a higher observed decline rate — directional, not causal.

## 7. Recommendation

The queue maps observable patterns to concrete review actions:

- **High visibility + low CTR** (1,473 test observations) → review title, snippet, and search-intent alignment.
- **High traffic + low engagement** (142 observations) → review content experience and intent match.
- **Demand with clicks** (2,561 observations) → review for a possible refresh opportunity.
- **Visible without a strong issue** → route to general human review.
- **Low evidence** (886 observations) → monitor only; do not escalate on probability alone.

**How a FlyRank editor uses this tomorrow:** open the ranked queue, start from rank 1, and work down. Each row shows the model probability, the evidence-aware priority score, the reason code, and the recommended action — enough context to triage without re-deriving anything. The evidence-aware layer specifically prevents sparse, low-impression observations with an inflated probability from crowding out well-evidenced opportunities at the top of the list.

**Confidence and limits, stated explicitly:** this is directional decision-support based on one March→April observation window and one validation split. It does not prove a refresh will work, does not model causal mechanisms, and should not be treated as an automatic publishing or content-change trigger. Any editor decision made from this queue is still a human decision.

## 8. Reproducibility

**Notebook:** `work/notebooks/capstone.ipynb` — runs top-to-bottom without errors.

**Re-run from a fresh clone:**
```
pip install pandas numpy scikit-learn huggingface_hub
# open work/notebooks/capstone.ipynb in Colab or Jupyter
# provide HF_TOKEN with access to FlyRank/internship-warehouse
# run all cells in order
```

**Random seeds:** `random_state=42` used consistently for both `GroupShuffleSplit` and `LogisticRegression`.

**Environment highlights:** pandas, numpy, scikit-learn (`GroupShuffleSplit`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `LogisticRegression`, `roc_auc_score`, `average_precision_score`, `brier_score_loss`, `accuracy_score`), `huggingface_hub` for dataset access.

**Outputs written on each run:**
- `work/outputs/capstone_model_vs_baseline.csv`
- `work/outputs/w08_content_action_queue.csv`
- `work/outputs/w08_archetype_action_summary.csv`
- `work/outputs/w08_monitoring_snapshot.csv`
- `work/outputs/w08_decay_refresh_summary.csv`

---
