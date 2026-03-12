"""Bootstrap v7 data files — run once after deploy."""
from src.hold_analyzer import rebuild_hold_analysis
from src.prediction_diagnosis import rebuild_prediction_diagnosis
from src.hypothesis_ledger import rebuild_hypothesis_ledger
from src.constraint_generator import generate_constraints
from src.pattern_explorer import rebuild_exploration_map

print("=== Hold Analysis ===")
ha = rebuild_hold_analysis()
print("Resolved:", ha.get("total_resolved", 0))
summary = ha.get("summary", {})
print("Quality:", summary.get("overall_hold_quality", 0))
print("HOLD_BETTER:", summary.get("conditions_where_hold_better", []))
print("TRADE_BETTER:", summary.get("conditions_where_trade_better", []))

print("\n=== Prediction Diagnosis ===")
pd = rebuild_prediction_diagnosis()
print("Analyzed:", pd.get("total_analyzed", 0))
insights = pd.get("prescriptive_insights", [])
for i in insights:
    print(" ", i)

print("\n=== Hypothesis Ledger ===")
rebuild_hypothesis_ledger()

print("\n=== Constraints ===")
c = generate_constraints()
print("Total:", c.get("total_constraints", 0))
for con in c.get("constraints", [])[:8]:
    print("  %s: %s — %s" % (con["type"], con["pattern"], con.get("reason", "")))

print("\n=== Exploration Map ===")
em = rebuild_exploration_map()
cov = em.get("tag_space_coverage", {})
print("Traded: %s/%s conditions" % (cov.get("traded", 0), cov.get("total_possible", 0)))
print("Held only:", cov.get("held_only", 0))
print("Unobserved:", cov.get("unobserved", 0))
print("Gaps:", len(em.get("exploration_gaps", [])))
candidates = em.get("ranked_candidates", [])
print("Candidates:", len(candidates))
for c in candidates[:3]:
    print("  %s (score=%.2f, holds=%d, trades=%d)" % (
        c["condition"], c["exploration_score"], c["hold_count"], c["trade_count"]))
