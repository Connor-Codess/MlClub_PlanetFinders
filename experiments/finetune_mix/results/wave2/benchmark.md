# lc-benchmarks: pt_mix-r00-deep-all against the published papers

| Benchmark | Paper | Their data | Their result | Ours (mean ± sd over seeds) | Test set covered |
|---|---|---|---|---|---|
| `astronet_dr24` | Shallue & Vanderburg 2018 (AstroNet) | Kepler DR24, PC vs AFP+NTP | AUC 0.988; accuracy 0.960 at threshold 0.5 (1,523 test TCEs) | AUC 0.860 ± 0.002; AP 0.608 ± 0.006; PC vs NTP AUC 0.931 ± 0.002; PC vs AFP AUC 0.841 ± 0.003 | 1,559 of 1,574 (99%) |
| `astronet_triage` | Yu et al. 2019 (Astronet-Triage) | TESS QLP sectors 1-5 | Triage (PC+EB vs J): AUC 0.992, AP 0.970. Vetting (PC vs rest): AUC 0.984, AP 0.693 | Triage: AUC 0.579 ± 0.021, AP 0.587 ± 0.010. Vetting: AUC 0.908 ± 0.016, AP 0.568 ± 0.012 | 660 of 1,658 (40%) |
| `astronet_triage_v2` | Tey et al. 2023 (Astronet-Triage-v2) | TESS QLP cycles 1-3, E vs rest | AUC-PR 0.965; recall 97.2% at precision 75.7% | AUC-PR 0.313 ± 0.005; AUC 0.425 ± 0.012 | 1,013 of 2,516 (40%) |
| `salinas2025` | Salinas et al. 2025 | TESS-SPOC FFI sectors 1-26 | AUC 0.88, F1 0.82 on an unreleased 10% split; 213 published candidates | Candidates detected: multi_planet_candidate 14.3%, periodic_candidate 74.0%, single_candidate 85.0%. AUC against TESS non-transit signals 0.968 ± 0.006 (single 0.986 ± 0.004, periodic 0.959 ± 0.005); against lc-eval random windows 0.961 ± 0.005 | 209 of 213 (98%) |

Detection rate by label at 1% random-window false alarms on lc-eval val:

| Benchmark | Label | Signals | Detected |
|---|---|---:|---:|
| `astronet_dr24` | AFP | 967 | 9.7% |
| `astronet_dr24` | NTP | 249 | 0.7% |
| `astronet_dr24` | PC | 343 | 38.8% |
| `astronet_triage` | EB | 139 | 12.7% |
| `astronet_triage` | J | 481 | 1.6% |
| `astronet_triage` | PC | 40 | 51.7% |
| `astronet_triage_v2` | B | 20 | 0.0% |
| `astronet_triage_v2` | E | 168 | 9.9% |
| `astronet_triage_v2` | J | 594 | 0.8% |
| `astronet_triage_v2` | N | 5 | 0.0% |
| `astronet_triage_v2` | S | 21 | 12.7% |
| `astronet_triage_v2` | none | 205 | 0.3% |
| `salinas2025` | multi_planet_candidate | 7 | 14.3% |
| `salinas2025` | periodic_candidate | 122 | 74.0% |
| `salinas2025` | single_candidate | 80 | 85.0% |
