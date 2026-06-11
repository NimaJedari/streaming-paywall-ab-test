Report : https://nimajedari.github.io/streaming-paywall-ab-test/

# A/B Test: Personalized Paywall Upsell — Premium Conversion

**Tools:** Python · SciPy · Pandas · NumPy · Survival Analysis · Stratified Randomization

> **Note:** Dataset is simulated. Real paywall experiment data is proprietary to streaming platforms. This project demonstrates the full analytical methodology used in production A/B testing environments at consumer tech companies.

---

## Business Question

Does anchoring a premium upsell modal to a user's own behavioral signals convert free users to paid subscribers at a higher rate than a generic static prompt?

---

## Experiment Design

| | |
|---|---|
| **Unit of randomization** | User-level |
| **Traffic split** | 50/50 |
| **Experiment window** | 30 days |
| **Sample size** | 12,000 users (6,000 per group) |
| **Primary metric** | 30-day premium conversion rate |
| **Secondary metrics** | Time-to-conversion, revenue per user |
| **Alpha** | 0.05 |
| **Power** | 80% |
| **MDE** | +3 percentage points absolute |

**Control** — Generic static modal shown identically to all users:
> *"Unlock Premium. Get unlimited access to all shows and movies for $9.99/month."*

**Treatment** — Modal dynamically constructed from the user's strongest behavioral signal:

| Signal | Message |
|---|---|
| High completion rate | *"You're 2 episodes from finishing Succession — don't stop now."* |
| Repeat visits | *"You've visited Severance 3 times this week. It's waiting for you."* |
| Large paywalled watchlist | *"8 titles on your watchlist are Premium. One sub unlocks all of them."* |
| Binge behavior | *"You watched 4 episodes in one sitting. You're exactly who Premium is made for."* |

---

## Methodology

Power analysis established a minimum of 1,774 users per group. Randomization used stratified sampling — users were first grouped by engagement tier (casual, regular, power), then split 50/50 within each stratum to guarantee balanced distributions by design.

Pre-analysis sanity checks validated randomization integrity:
- Sample Ratio Mismatch (SRM) test
- Tenure balance (t-test)
- Device distribution (chi-square)
- Engagement tier balance (chi-square)

Primary analysis used a two-proportion z-test. Time-to-conversion was modeled using a **Kaplan-Meier survival curve implemented from scratch** in Python (scipy/numpy), with a **log-rank test** to compare curves. Subgroup analysis was conducted by engagement tier and behavioral signal type.

---

## Results

| Metric | Control | Treatment | Lift |
|---|---|---|---|
| Conversion rate | 9.33% | 16.68% | **+7.35pp** |
| Median time-to-convert | Day 7 | Day 4 | **3 days faster** |
| Revenue per user | $1.81 | $3.18 | **+$1.37** |

- **Primary test:** Z = 11.96, p < 0.0001, 95% CI [+6.1pp, +8.5pp] ✓
- **Log-rank test:** chi2 = 147.2, p < 0.0001 ✓
- The conversion gap opened within the first week and remained stable through Day 30

**Subgroup: Conversion by engagement tier**

| Tier | Control | Treatment | Lift |
|---|---|---|---|
| Casual | 6.4% | 12.1% | +5.7pp |
| Regular | 10.3% | 18.1% | +7.8pp |
| Power | 17.3% | 29.3% | +12.0pp |

**Subgroup: Lift by behavioral signal (treatment only)**

| Signal | Lift vs. Control |
|---|---|
| Binge behavior | +10.1pp |
| High completion rate | +8.7pp |
| Repeat visits | +6.8pp |
| Large watchlist | +4.1pp |

**Projected business impact:** ~$70M incremental annual revenue (20M MAU, 40% paywall exposure, $9.99/month ARPU)

---

## Recommendation

**Ship to 100% of eligible users.**

Priority follow-up: run a dedicated experiment on mid-session binge-triggered timing — showing the modal at the natural pause point between episodes — to test whether the binge signal can be activated even more effectively at the moment of highest intent.

---

## Repository Structure

```
paywall_ab_test/
│
├── notebooks/
│   └── ab_test_paywall.py       # Full analysis script
│
├── outputs/
│   ├── summary.csv              # Top-level results by variant
│   ├── km_curves.csv            # Kaplan-Meier survival estimates (day 0–30)
│   ├── subgroup_tier.csv        # Conversion rates by engagement tier
│   └── subgroup_signal.csv      # Conversion rates by behavioral signal
│
└── README.md
```

---

## How to Run

```bash
# Requirements: Python 3.8+, numpy, pandas, scipy
pip install numpy pandas scipy

python notebooks/ab_test_paywall.py
```

No external data files needed — the simulation generates all data on run.

---

*Author: Nima Jedari · [LinkedIn](https://linkedin.com/in/nimajedari)*
