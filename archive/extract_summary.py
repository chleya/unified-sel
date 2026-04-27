import json, sys
d = json.load(open(sys.argv[1]))
print(f"ewc_lambda = {d.get('ewc_lambda', 0)}")
print(f"{'Seed':>6} | {'T0_final':>8} | {'T1_final':>8} | {'forget':>8} | {'avg_acc':>8}")
print("-" * 55)
for r in d['runs']:
    t0 = r['task_0_accuracy_final']
    t1 = r['task_1_accuracy_final']
    fg = r['forgetting_task_0']
    avg = (t0 + t1) / 2
    print(f"{r['seed']:>6} | {t0:>8.4f} | {t1:>8.4f} | {fg:>8.4f} | {avg:>8.4f}")
s = d['summary']
print(f"\nMean: T0={s['task_0_accuracy_final']['mean']:.4f} T1={s['task_1_accuracy_final']['mean']:.4f} forget={s['forgetting_task_0']['mean']:.4f}")
