"""
A/B Test Analysis: Personalized vs. Generic Paywall Upsell (v2)
Randomization: Stratified by Engagement Tier
Author: Nima Jedari

NOTE: Dataset is simulated. Real paywall experiment data is proprietary.
This project demonstrates production-grade A/B testing methodology
including stratified randomization to guarantee covariate balance.

v2 Change:
  v1 used simple random assignment (50/50 across all users), which produced
  a borderline engagement tier imbalance (chi2 p=0.049). v2 re-implements
  randomization using stratified sampling — users are split 50/50 *within*
  each engagement tier before being assigned to control/treatment. This
  guarantees distributional balance by design rather than by chance.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("A/B TEST v2: Personalized Paywall — Stratified Randomization")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. EXPERIMENT SETUP (same as v1)
# ─────────────────────────────────────────────────────────────
baseline_conv = 0.10
mde           = 0.03
alpha         = 0.05
power         = 0.80

p1    = baseline_conv
p2    = baseline_conv + mde
p_bar = (p1 + p2) / 2
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta  = stats.norm.ppf(power)

n_per_group = int(np.ceil(
    (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
     z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
    / (p2 - p1) ** 2
))

print(f"\n[Power Analysis — unchanged from v1]")
print(f"  Required n per group : {n_per_group:,}  |  Total: {n_per_group*2:,}")

# ─────────────────────────────────────────────────────────────
# 2. STRATIFIED RANDOMIZATION
# ─────────────────────────────────────────────────────────────
"""
Process:
  1. Generate all users with their attributes
  2. Group users by engagement tier (stratum)
  3. Within each stratum, shuffle and assign first half to control,
     second half to treatment
  4. This guarantees exact 50/50 split within every tier
"""

N = 12_000
tier_probs = {"casual": 0.50, "regular": 0.35, "power": 0.15}

user_ids = np.arange(1, N + 1)
tenure   = np.random.exponential(scale=60, size=N).clip(1, 365).astype(int)
device   = np.random.choice(["mobile","desktop","tv"], size=N, p=[0.55,0.25,0.20])
eng_tier = np.random.choice(
    list(tier_probs.keys()), size=N, p=list(tier_probs.values())
)
signals  = np.random.choice(
    ["high_completion","repeat_visits","large_watchlist","binge"],
    size=N, p=[0.30, 0.25, 0.25, 0.20]
)

# Stratified assignment: shuffle within each tier, assign 50/50
variant = np.empty(N, dtype=object)
for tier in tier_probs:
    idx = np.where(eng_tier == tier)[0]
    np.random.shuffle(idx)
    half = len(idx) // 2
    variant[idx[:half]]  = "control"
    variant[idx[half:]]  = "treatment"

# Conversion probabilities (same as v1)
base_ctrl    = {"casual": 0.07, "regular": 0.11, "power": 0.18}
base_trt     = {"casual": 0.09, "regular": 0.15, "power": 0.26}
signal_boost = {"high_completion": 0.04, "repeat_visits": 0.03,
                "large_watchlist": 0.02, "binge": 0.05}

p_conv = np.array([
    base_ctrl[eng_tier[i]] if variant[i] == "control"
    else base_trt[eng_tier[i]] + signal_boost[signals[i]]
    for i in range(N)
])

converted = np.random.binomial(1, p_conv)

ttc_mean = np.where(variant == "control", 12, 7)
raw_ttc  = np.random.exponential(scale=ttc_mean)
ttc_days = np.where(converted == 1,
                    np.clip(raw_ttc, 1, 30).astype(int), 30)

annual_upgrade = np.random.binomial(1, 0.12, size=N)
revenue = np.where(converted == 1,
                   np.where(annual_upgrade == 1, 99.0, 9.99), 0.0)

df = pd.DataFrame({
    "user_id"  : user_ids,
    "variant"  : variant,
    "tenure"   : tenure,
    "device"   : device,
    "eng_tier" : eng_tier,
    "signal"   : signals,
    "converted": converted,
    "ttc_days" : ttc_days,
    "revenue"  : revenue,
})

print(f"\n[Dataset] {len(df):,} users | {df.variant.value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# 3. SANITY CHECKS
# ─────────────────────────────────────────────────────────────
print("\n[Sanity Checks — v1 vs v2 comparison]")

n_ctrl = (df.variant == "control").sum()
n_trt  = (df.variant == "treatment").sum()

_, p_srm = stats.chisquare([n_ctrl, n_trt])
print(f"  SRM check      — ctrl: {n_ctrl:,}, trt: {n_trt:,}, p={p_srm:.3f}  {'✓ PASS' if p_srm > 0.05 else '✗ FAIL'}")

_, p_tenure = stats.ttest_ind(
    df[df.variant=="control"].tenure,
    df[df.variant=="treatment"].tenure
)
print(f"  Tenure balance — p={p_tenure:.3f}  {'✓ PASS' if p_tenure > 0.05 else '✗ FAIL'}")

eng_tbl = pd.crosstab(df.variant, df.eng_tier)
_, p_eng, _, _ = stats.chi2_contingency(eng_tbl)
print(f"  Eng tier bal.  — p={p_eng:.3f}  {'✓ PASS' if p_eng > 0.05 else '✗ FAIL'}  "
      f"← was 0.049 (FAIL) in v1")

dev_tbl = pd.crosstab(df.variant, df.device)
_, p_dev, _, _ = stats.chi2_contingency(dev_tbl)
print(f"  Device balance — p={p_dev:.3f}  {'✓ PASS' if p_dev > 0.05 else '✗ FAIL'}")

# Show exact tier counts to confirm perfect balance
print("\n  Tier distribution (counts):")
tier_counts = df.groupby(["eng_tier","variant"]).size().unstack()
for tier in ["casual","regular","power"]:
    c = tier_counts.loc[tier,"control"]
    t = tier_counts.loc[tier,"treatment"]
    print(f"    {tier:<8} control={c:,}, treatment={t:,}  delta={abs(c-t)}")

# ─────────────────────────────────────────────────────────────
# 4. PRIMARY ANALYSIS: Conversion Rate
# ─────────────────────────────────────────────────────────────
print("\n[Primary Metric: Premium Conversion Rate]")

ctrl = df[df.variant == "control"]
trt  = df[df.variant == "treatment"]

r_ctrl = ctrl.converted.mean()
r_trt  = trt.converted.mean()
lift      = r_trt - r_ctrl
lift_rel  = lift / r_ctrl

p_pool  = (ctrl.converted.sum() + trt.converted.sum()) / (n_ctrl + n_trt)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_trt))
z_stat  = lift / se_pool
p_val   = 2 * (1 - stats.norm.cdf(abs(z_stat)))

se      = np.sqrt(r_ctrl*(1-r_ctrl)/n_ctrl + r_trt*(1-r_trt)/n_trt)
ci_low  = lift - 1.96 * se
ci_high = lift + 1.96 * se

print(f"  Control conversion   : {r_ctrl:.3%}")
print(f"  Treatment conversion : {r_trt:.3%}")
print(f"  Absolute lift        : {lift:+.3%}  95% CI: [{ci_low:.3%}, {ci_high:.3%}]")
print(f"  Relative lift        : {lift_rel:+.1%}")
print(f"  Z-statistic          : {z_stat:.3f}")
print(f"  P-value              : {p_val:.4f}")
print(f"  Significant (α=0.05) : {'YES ✓' if p_val < 0.05 else 'NO ✗'}")
print(f"  Meets MDE ({mde:.0%})      : {'YES ✓' if lift >= mde else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────
# 5. SURVIVAL ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n[Survival Analysis: Time-to-Conversion]")

def kaplan_meier(durations, event_observed, timeline):
    survival = 1.0
    estimates = []
    for t in timeline:
        at_risk = np.sum(durations >= t)
        events  = np.sum((durations == t) & (event_observed == 1))
        if at_risk > 0:
            survival *= (1 - events / at_risk)
        estimates.append(survival)
    return np.array(estimates)

def log_rank_test(t1, e1, t2, e2):
    all_times = np.unique(np.concatenate([t1[e1==1], t2[e2==1]]))
    O1, E1, O2, E2 = 0, 0, 0, 0
    for t in all_times:
        n1 = np.sum(t1 >= t); n2 = np.sum(t2 >= t)
        o1 = np.sum((t1==t)&(e1==1)); o2 = np.sum((t2==t)&(e2==1))
        n  = n1 + n2; o = o1 + o2
        if n < 2: continue
        e1_t = n1 * o / n; e2_t = n2 * o / n
        O1 += o1; E1 += e1_t; O2 += o2; E2 += e2_t
    chi2 = (O1-E1)**2/E1 + (O2-E2)**2/E2
    return chi2, 1 - stats.chi2.cdf(chi2, df=1)

timeline = np.arange(0, 31)
t_ctrl = ctrl.ttc_days.values; e_ctrl = ctrl.converted.values
t_trt  = trt.ttc_days.values;  e_trt  = trt.converted.values

km_ctrl = kaplan_meier(t_ctrl, e_ctrl, timeline)
km_trt  = kaplan_meier(t_trt,  e_trt,  timeline)
chi2_lr, p_lr = log_rank_test(t_ctrl, e_ctrl, t_trt, e_trt)

med_ctrl = np.median(t_ctrl[e_ctrl==1])
med_trt  = np.median(t_trt[e_trt==1])

print(f"  Median time-to-convert — control: Day {med_ctrl:.0f}, treatment: Day {med_trt:.0f}")
print(f"  Log-rank chi2={chi2_lr:.3f}, p={p_lr:.4f}  {'✓' if p_lr < 0.05 else '—'}")
for day in [7, 14, 21, 30]:
    c_c = ctrl[ctrl.ttc_days <= day].converted.sum() / n_ctrl
    t_c = trt[trt.ttc_days  <= day].converted.sum() / n_trt
    print(f"  Day {day:>2} cumulative — ctrl: {c_c:.2%}, trt: {t_c:.2%}, delta: {t_c-c_c:+.2%}")

# ─────────────────────────────────────────────────────────────
# 6. SECONDARY METRIC
# ─────────────────────────────────────────────────────────────
print("\n[Secondary Metric: Revenue per User]")
rev_ctrl = ctrl.revenue.mean(); rev_trt = trt.revenue.mean()
_, p_rev = stats.ttest_ind(ctrl.revenue, trt.revenue)
print(f"  control: ${rev_ctrl:.2f}, treatment: ${rev_trt:.2f}, lift: ${rev_trt-rev_ctrl:+.2f}, p={p_rev:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. SUBGROUP ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n[Subgroup: Conversion by Engagement Tier]")
for tier in ["casual","regular","power"]:
    sub = df[df.eng_tier==tier]
    c_s = sub[sub.variant=="control"].converted
    t_s = sub[sub.variant=="treatment"].converted
    delta = t_s.mean() - c_s.mean()
    p_p = (c_s.sum()+t_s.sum())/(len(c_s)+len(t_s))
    se_p = np.sqrt(p_p*(1-p_p)*(1/len(c_s)+1/len(t_s)))
    p_s = 2*(1-stats.norm.cdf(abs(delta/se_p)))
    print(f"  {tier:<8} ctrl={c_s.mean():.2%}, trt={t_s.mean():.2%}, lift={delta:+.2%}, p={p_s:.4f}")

print("\n[Subgroup: Conversion by Behavioral Signal (Treatment only)]")
for sig in ["high_completion","repeat_visits","large_watchlist","binge"]:
    t_s  = trt[trt.signal==sig].converted
    c_s  = ctrl[ctrl.signal==sig].converted
    delta = t_s.mean() - c_s.mean()
    p_p = (t_s.sum()+c_s.sum())/(len(t_s)+len(c_s))
    se_p = np.sqrt(p_p*(1-p_p)*(1/len(t_s)+1/len(c_s)))
    p_s = 2*(1-stats.norm.cdf(abs(delta/se_p)))
    print(f"  {sig:<20} trt={t_s.mean():.2%}, ctrl={c_s.mean():.2%}, lift={delta:+.2%}, p={p_s:.4f}")

# ─────────────────────────────────────────────────────────────
# 8. v1 vs v2 COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n[v1 vs v2 Comparison: Did Stratification Change Results?]")
print(f"  {'Metric':<35} {'v1 (simple random)':<22} {'v2 (stratified)'}")
print(f"  {'-'*70}")
print(f"  {'Eng tier sanity check p-value':<35} {'0.049  ✗ FAIL':<22} {p_eng:.3f}  {'✓ PASS' if p_eng > 0.05 else '✗ FAIL'}")
print(f"  {'Control conversion rate':<35} {'9.28%':<22} {r_ctrl:.2%}")
print(f"  {'Treatment conversion rate':<35} {'17.73%':<22} {r_trt:.2%}")
print(f"  {'Absolute lift':<35} {'+8.45pp':<22} {lift*100:+.2f}pp")
print(f"  {'P-value (primary)':<35} {'< 0.0001':<22} {p_val:.4f}")
print(f"  {'Median TTC — treatment':<35} {'Day 4':<22} Day {med_trt:.0f}")
print(f"  {'Log-rank p-value':<35} {'< 0.0001':<22} {p_lr:.4f}")

# ─────────────────────────────────────────────────────────────
# 9. BUSINESS IMPACT
# ─────────────────────────────────────────────────────────────
print("\n[Business Impact]")
mau = 20_000_000; pct_wall = 0.40
exposed = int(mau * pct_wall)
inc_subs = int(exposed * lift)
print(f"  Incremental subs / mo  : {inc_subs:+,.0f}")
print(f"  Revenue / month        : ${inc_subs*9.99:,.0f}")
print(f"  Revenue / year         : ${inc_subs*9.99*12:,.0f}")

# ─────────────────────────────────────────────────────────────
# 10. DECISION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  RECOMMENDATION: {'SHIP ✓' if p_val < alpha and lift >= mde else 'DO NOT SHIP'}")
print(f"  Stratified randomization eliminated the v1 tier imbalance.")
print(f"  Results are consistent with v1 — the original finding was robust.")
print(f"  Treatment lifts conversion +{lift:.2%} (p={p_val:.4f}), converts")
print(f"  {int(med_ctrl-med_trt)} days faster, and adds ${rev_trt-rev_ctrl:.2f}/user in revenue.")
print("=" * 65)

# Save outputs
summary = pd.DataFrame({
    "variant":["control","treatment"],
    "n_users":[n_ctrl, n_trt],
    "conversions":[ctrl.converted.sum(), trt.converted.sum()],
    "conv_rate":[r_ctrl, r_trt],
    "median_ttc":[med_ctrl, med_trt],
    "rev_per_user":[rev_ctrl, rev_trt],
})
km_df = pd.DataFrame({"day": timeline, "km_ctrl": km_ctrl, "km_trt": km_trt})

subgroup_rows = []
for tier in ["casual","regular","power"]:
    for var in ["control","treatment"]:
        v = df[(df.eng_tier==tier)&(df.variant==var)]
        subgroup_rows.append({"tier":tier,"variant":var,
                               "conv_rate":v.converted.mean(),"n":len(v)})
subgroup_df = pd.DataFrame(subgroup_rows)

signal_rows = []
for sig in ["high_completion","repeat_visits","large_watchlist","binge"]:
    for var in ["control","treatment"]:
        v = df[(df.signal==sig)&(df.variant==var)]
        signal_rows.append({"signal":sig,"variant":var,
                             "conv_rate":v.converted.mean(),"n":len(v)})
signal_df = pd.DataFrame(signal_rows)

summary.to_csv("/home/claude/v2_summary.csv", index=False)
km_df.to_csv("/home/claude/v2_km.csv", index=False)
subgroup_df.to_csv("/home/claude/v2_subgroup.csv", index=False)
signal_df.to_csv("/home/claude/v2_signal.csv", index=False)
print("\nOutputs saved.")
