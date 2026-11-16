---
license: other
pretty_name: TESS Objects of Interest with TFOPWG dispositions (2026-11-09 snapshot)
tags:
- astronomy
- exoplanets
- tess
configs:
- config_name: default
  data_files:
  - split: train
    path: "TOI_2026.11.09_18.12.19.csv"
  comment: "#"
---

# TESS Objects of Interest with TFOPWG dispositions (2026-11-09 snapshot)

A four-column export of the TESS Object of Interest (TOI) table from the NASA Exoplanet Archive, downloaded
2026-11-09 18:12 UTC. 8,151 TOIs.

| Column | Description |
|---|---|
| `toi` | TOI number (for example 1001.01) |
| `toipfx` | TOI prefix: the host system's number |
| `tid` | TESS Input Catalog ID |
| `tfopwg_disp` | TESS Follow-up Observing Program Working Group disposition |

| Disposition | Meaning | Count |
|---|---|---:|
| PC | Planet candidate | 4,813 |
| FP | False positive | 1,319 |
| CP | Confirmed planet | 819 |
| KP | Known planet | 606 |
| APC | Ambiguous planet candidate | 494 |
| FA | False alarm | 98 |
| blank | No disposition | 2 |

The file starts with `#` comment lines from the archive; pass `comment="#"` when reading it with pandas.

**Superseded.** For periods, epochs, durations, depths and stellar parameters use `toi_ephemerides` in
[reference-catalogs](https://huggingface.co/datasets/nyu-planet-finders/reference-catalogs), which is rebuilt
from the archive every month. This snapshot is kept so earlier work can be reproduced.

## Source and acknowledgment

NASA Exoplanet Archive TOI table (Guerrero et al. 2021, ApJS 254, 39). This research has made use of the NASA
Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the
National Aeronautics and Space Administration under the Exoplanet Exploration Program.
