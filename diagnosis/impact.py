"""Business impact translation (architecture 4.5).

Honest by construction: GMV at risk is an explicit upper bound (all non-success
transactions in the window, regardless of cause), and the manual-analyst baseline
is a single documented constant, not a made-up per-case number.
"""

from __future__ import annotations

# Documented manual-analyst baseline: dashboards + slicing to reach a confident
# root cause. Kept as one explicit constant so the claim is auditable.
MANUAL_BASELINE_MINUTES = 45.0


def estimate_impact(con, current_start, current_end,
                     elapsed_minutes: float) -> dict:
    row = con.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(amount), 0.0)
        FROM transactions
        WHERE ts >= ? AND ts < ? AND status != 'success'
        """,
        [current_start, current_end],
    ).fetchone()
    n_non_success, gmv_at_risk = int(row[0]), float(row[1])
    window_minutes = (current_end - current_start).total_seconds() / 60
    # round the published intermediates first, then derive the headline number
    # from them: keeps reports deterministic and internally consistent
    hours_saved = round(max(0.0, (MANUAL_BASELINE_MINUTES - elapsed_minutes) / 60), 2)
    gmv_per_hour = round(gmv_at_risk / max(window_minutes / 60, 1e-9), 2)
    return {
        "manual_baseline_minutes": MANUAL_BASELINE_MINUTES,
        "time_to_diagnosis_minutes": round(elapsed_minutes, 2),
        "hours_saved_vs_manual": round(hours_saved, 2),
        "window_minutes": round(window_minutes, 1),
        "non_success_txns_in_window": n_non_success,
        "gmv_at_risk_inr": round(gmv_at_risk, 2),
        "gmv_at_risk_per_hour_inr": gmv_per_hour,
        "gmv_protected_estimate_inr": round(gmv_per_hour * hours_saved, 2),
        "note": ("GMV at risk is an upper bound: all non-success transactions in the "
                 "window regardless of cause. gmv_protected assumes mitigation at "
                 "diagnosis time."),
    }
