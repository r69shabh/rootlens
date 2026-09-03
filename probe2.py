import sys, numpy, duckdb
print("python", sys.version.split()[0], "numpy", numpy.__version__, "duckdb", duckdb.__version__)
from data_engine.generator import TransactionGenerator, WindowConfig
from data_engine.faults import HighTicketRuleInjector, BankOutageInjector
from datetime import datetime, timezone
from diagnosis.anomaly_scan import scan, _significant
wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=timezone.utc))
S, E, B = wc.current_window_start, wc.current_window_end, wc.start

con = TransactionGenerator(seed=42, window=wc, txns_per_day=4000).generate()
n10 = con.execute("SELECT COUNT(*) FROM transactions WHERE amount > 10000 AND ts >= ? AND ts < ?", [S, E]).fetchone()[0]
n10h2 = con.execute("SELECT COUNT(*) FROM transactions WHERE amount > 10000 AND ts >= ? AND ts < ?", [S + (E-S)/2, E]).fetchone()[0]
print("high_ticket: >10k in window:", n10, "in second half:", n10h2)
HighTicketRuleInjector(wc, 10000).inject(con)
segs = scan(con, S, E, B, S)
print("high_ticket segments:", [(s.dimension, s.value, round(s.drop,3), s.current_volume) for s in segs[:5]])
row = con.execute("""SELECT COUNT(*), AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END)
    FROM transactions WHERE amount > 10000 AND ts >= ? AND ts < ?""", [S, E]).fetchone()
print(">10k bucket n, success_rate:", row)

con2 = TransactionGenerator(seed=42, window=wc, txns_per_day=4000).generate()
BankOutageInjector(wc, "ICICI").inject(con2)
rows = con2.execute("""SELECT issuer_bank, COUNT(*), AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END)
    FROM transactions WHERE ts >= ? AND ts < ? GROUP BY 1""", [S, E]).fetchall()
print("bank matrix (n, success_rate):", [(r[0], r[1], round(r[2],3)) for r in rows])
segs2 = scan(con2, S, E, B, S)
print("bank_outage segments:", [(s.dimension, s.value, round(s.drop,3)) for s in segs2[:4]])
