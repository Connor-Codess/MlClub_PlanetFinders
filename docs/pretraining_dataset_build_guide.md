# Pretraining dataset build guide

> Exported from the team's Claude doc "Pretraining dataset build guide" (https://claude.ai/artifact/3q2VWUSb4UxhiQoQZ6Hyqr) on 2026-11-15. The live doc has the charts and diagrams; this copy keeps the text and tables.


Nov 14, 2026 · @Connor

## Overview and what is on Hugging Face

Build one canonical pretraining store: every light curve from all three missions on a single 30-minute time grid, cleaned, normalized and tagged once, so that any transformer can read it. The build reads six existing repos and writes one new one. Kepler q15-q17 and TOI ephemerides should be fixed first (Step 0).

**Design rule.** Bake in only what is expensive to redo or must be identical for every model: cleaning, the time grid, normalization statistics, splits and exclusions. Leave everything you will want to test to the data loader at training time: window length, flux compression, which channels a model sees, patching, masking and the source mix. One build then serves every proxy experiment.

### Inventory (checked 2026-11-14 on the Hub)

| Repo | Contents | Rows | Size | Status |
| --- | --- | --- | --- | --- |
| [tess-spoc-ffi-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-ffi-binned) | TESS-SPOC FFI, 30-min bins, sectors 1-81; tic\_id, sector, tmag, teff, radius, time, flux, flux\_err, centr\_x, centr\_y, bkg, n\_points | 12.8M star-sectors | 252.8 GB | Complete (81 files) |
| [tess-spoc-2min-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-2min-binned) | SPOC 2-min binned to 30 min, sectors 1-107; same columns minus bkg and n\_points | 1,734,057 | 27.7 GB | Complete. Now public (the handoff doc says private) |
| [kepler-llc](https://huggingface.co/datasets/nyu-planet-finders/kepler-llc) | Kepler long cadence, native 29.4 min, one file per quarter; kic\_id, quarter, kepmag, teff, radius, time, flux, flux\_err, centr\_x, centr\_y | 2.4M star-quarters | 166.4 GB | q00-q14 only. The manifest lists 2.9M files, so about 0.5M star-quarters (q15-q17) are missing |
| [kepler-raw-lightcurves](https://huggingface.co/datasets/nyu-planet-finders/kepler-raw-lightcurves) | Unbinned DR25 light curves for the 9,865 DR24 TCE stars, q0-q17, with sap\_quality; plus the DR24 TCE table (20,367 TCEs, PC/AFP/NTP labels) | 158,198 | 10.5 GB | Complete. Same stars as kepler-llc, so no new pretraining stars |
| [kepler-centroids](https://huggingface.co/datasets/nyu-planet-finders/kepler-centroids) | mom\_centr1/2 and pos\_corr1/2, row-aligned with kepler-raw-lightcurves | 158,198 | \~12 GB | Complete |
| [TESS\_TCES](https://huggingface.co/datasets/nyu-planet-finders/TESS_TCES) | TOI list from the NASA Exoplanet Archive, 2026-11-09 | one CSV | 209 KB | Only 4 columns: toi, toipfx, tid, tfopwg\_disp. No periods, epochs or durations |

### What the inventory means for the build

- **The sources disagree in format.** FFI has background and points-per-bin; 2-min and Kepler do not. FFI keeps rows with missing centroids; the others dropped them. 2-min bins start at each light curve's first good cadence, so bins are not aligned across stars. Steps 4 and 6 make these uniform.
- **Quality filtering cannot be harmonized.** FFI used lightkurve's default bitmask (17087); 2-min and Kepler used QUALITY == 0. The flags were not stored, so the build records the policy per row instead of re-filtering.
- **Real data confirms the noise choice.** In one FFI row (TIC 29257288, sector 1, an M dwarf at Tmag 11.8), point-to-point noise is 1,190 ppm but a plain MAD gives 4,330 ppm, because the star varies. Step 5 uses the point-to-point estimate.
- **I could only inspect metadata and one row.** This session's shell cannot reach huggingface.co, so column ranges for centroid, background and n\_points come from the handoff doc, not from reading them. Step 11 checks them on the real data.

## What the finished dataset looks like

One row is one **segment**: one star observed for one TESS sector or one Kepler quarter, placed on the shared 30-minute grid. A TESS segment is about 1,330 slots and a Kepler quarter about 4,400. A model never reads a row directly; the loader cuts windows out of it.

&#91;embedded content: tess-spoc-ffi-binned s0001.parquet, first row · every second observed slot plotted\]

This M dwarf repeats a smooth swing every \~10.6 hours, which is why noise is measured point to point: a plain MAD would call this star 3.6 times noisier than it is. The 32 slots past the default window stay in the row, so a loader can use another window length or a random offset.

### Anatomy of one row

| Group | Columns | Stored as | Example (the segment above) |
| --- | --- | --- | --- |
| Identity | segment\_id, star\_key, tic\_id, kic\_id, source, sector or quarter, cycle | strings, ints | `ffi:s0001:29257288`, `tic:29257288`, ffi, 1 |
| Grid | slot0, n\_slots, n\_obs, coverage | ints, float | n\_slots 1,328; n\_obs 1,176; coverage 0.886 |
| Per-slot channels | flux\_z, flux\_err\_z, cen\_x, cen\_y, bkg\_z (float16); fill (uint8); flags (uint8 bits) | one byte array per channel, length n\_slots | 12 bytes per slot |
| Normalization | flux\_median, sigma\_p2p\_ppm, sigma\_mad\_ppm, var\_ratio, bkg\_sigma | floats | 1,190 ppm; 4,330 ppm; 3.6 |
| Star | tmag, kepmag, teff, radius, missing flags | floats, bits | Tmag 11.85; Teff 3,258 K; 0.247 R☉ |
| Tags | toi, toi\_disposition, kepler\_tce\_label, eb\_catalog, dup\_group, quality\_tier | strings | none for this star |
| Split and weights | hash\_bucket, split, w\_source, w\_star, w\_var | int, string, floats | set by Steps 1 and 9 |
| Provenance | source\_repo, source\_file, source\_row, build\_version | strings | tess-spoc-ffi-binned, s0001.parquet, row 0 |

At 12 bytes per slot a TESS segment is about 16 KB and a Kepler quarter about 53 KB before compression. The full build comes to roughly 370 GB, less than the 447 GB of source repos it reads, because it drops the float64 time arrays and stores float16.

## Step 0: fix the gaps before building

Finish Kepler q15-q17 and download four small catalogs before writing any rows. Everything later depends on them: splits need the KIC-to-TIC crossmatch, and exclusions and tags need the catalogs.

**Technical details**

1. **Kepler q15-q17.** Upload any q15/q16 parquet still in TIKE, then run kepler\_binned.py for the rest, as handoff item 2 describes. Check: the light\_curves config should reach about 2.9M rows, matching the manifest's 2.9M files.
2. **A TOI table with ephemerides.** The TESS\_TCES file has only TOI, TIC and disposition. Re-export it from the NASA Exoplanet Archive TOI table with: toi, tid, tfopwg\_disp, pl\_orbper, pl\_tranmid, pl\_trandurh, pl\_trandep, pl\_rade, st\_tmag, plus sectors if offered. TIC-only exclusions work without this, but choosing "long-period" test planets does not.
3. **Eclipsing-binary catalogs.** The TESS EB catalog (Prsa et al. 2022) and the Kepler EB catalog (Kirk et al. 2016), keyed by TIC and KIC. Used only as tags.
4. **Duotransit list.** Host TIC IDs from the candidate table in Hawthorn et al. 2024. These become test stars.
5. **KIC-to-TIC crossmatch.** The TIC v8 catalog carries a KIC cross-identifier. Pull it for every KIC in the kepler-llc manifest (about 200k IDs) from MAST's TIC catalog query, and store kic\_id to tic\_id as a small parquet. Some Kepler-field stars were also observed by TESS, and this is how they end up in the same split.

**Decision 0: what to do about Kepler q15-q17**

| Option | What you get | Cost |
| --- | --- | --- |
| **A. Finish them first (recommended)** | All 18 quarters, one consistent reduction | One more resumable TIKE run |
| B. Build now, add them as build v1.1 | Start sooner; about 17% of Kepler star-quarters missing at first | A second build pass over Kepler later |
| C. Fill only DR24 stars from kepler-raw-lightcurves | 9,865 stars get q15-q17 now | Those rows need the same reduction as kepler\_binned.py, or they differ from the rest |

Recommendation: A. The pipeline already exists and is resumable, and a single consistent reduction avoids a second version of the Kepler rows.

**Why kepler-raw-lightcurves and kepler-centroids are not inputs here.** They cover the same 9,865 stars that kepler-llc already contains. Keep them for supervised work: the DR24 labels, phase-folded views and centroid vetting.

## Step 1: give every star one identity and one split

Assign each star a single key and a hash bucket from 0 to 999; the bucket decides its split forever. Splitting by star rather than by row means a star's 2-min and FFI rows, and a Kepler star later seen by TESS, can never land on both sides of a split.

**Technical details**

- **star\_key** = `tic:<TIC ID>` for every TESS row, and for Kepler stars that have a TIC match from Step 0. Kepler stars without a match get `kic:<KIC ID>`.
- **hash\_bucket** = SHA-256 of the star\_key string, first 8 bytes read as an unsigned integer, modulo 1000. Any language reproduces it from the key alone, so no lookup table has to travel with the data.
- Store a small **star table** (one row per star\_key: tic\_id, kic\_id, bucket, split, number of segments per source). Steps 2, 9 and 11 read it.

**Recommended bucket layout**

| Buckets | Share | Split | Used for |
| --- | --- | --- | --- |
| 0-849 | 85% | pretrain-train | Pretraining |
| 850-899 | 5% | pretrain-val | Pretraining validation loss, early stopping |
| 900-949 | 5% | search-holdout | Searched like every other star but never pretrained on: the control group for the seen-versus-unseen bias check |
| 950-999 | 5% | scoreboard-hosts | Host light curves for injected validation and test transits; never pretrained on; their real light curves are still searched |

**Decision 1: how to split**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Hash buckets on star\_key (recommended)** | Reproducible from the key; duplicates follow automatically; no table to keep in sync | Validation set is a random draw, not balanced by magnitude |
| B. Stratified by source and Tmag bin | Every magnitude range equally represented in validation | Needs a stored assignment table; a new data release can reshuffle stars |
| C. By sky region or camera | Tests whether the model generalizes to unseen parts of the sky | Correlated with systematics in ways that complicate the scoreboard |

Recommendation: A. A random 5% of millions of stars already covers every magnitude range; if a range turns out thin, take more hosts from it inside buckets 950-999 rather than changing the scheme.

If 15% held out of pretraining feels like too much, shrink search-holdout to 2% (buckets 900-919) and return the rest to pretrain-train. Decide before the build; changing it later means rebuilding.

## Step 2: lock the exclusion list

Remove every star the scoreboard will use from the pretraining build, and write their rows to a separate held-out repo. If those stars are physically absent from the pretraining repo, no loader, notebook or teammate can leak them by accident.

**Technical details**

1. Start from buckets 900-999 (search-holdout and scoreboard-hosts, Step 1).
2. Add list exclusions by star\_key, whatever their bucket:
   - every Hawthorn et al. 2024 duotransit host;
   - every TOI dispositioned PC, APC, CP or KP whose period is above 20 days or blank (single-transit TOIs often have no period). Split these by hash into a real-validation half and a real-test half. This needs the ephemeris export from Step 0.
3. Translate the list through the KIC-to-TIC crossmatch, so a held-out TESS star is also removed from Kepler if it was observed there.
4. Save `exclusions.parquet`: star\_key, reason (bucket, duotransit, long-period TOI), and which scoreboard set it feeds.
5. In every later step, a row whose star\_key is in this file goes to the **heldout** output, never to **pretrain**.

**Decision 2: how much to exclude**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Scoreboard stars only (recommended)** | Pretraining still sees thousands of real transits and EBs from other TOIs, which is useful "abnormal" data | Discovery claims rest on the seen-versus-unseen check (search-holdout) |
| B. Also exclude every TOI and catalogued EB star | Cleanest story for a paper | Removes most real transits and eclipses from pretraining, the events the model most needs to see |
| C. Exclude nothing beyond buckets | Simplest | Real held-out planets leak into pretraining and inflate test scores |

Recommendation: A, with the 20-day cut. If the team later wants B, it is a filter on the eb\_catalog and toi tags (Step 7) at load time, so it needs no rebuild.

**This step freezes your test set.** Any star you might later want for testing has to be added now. Adding a test star after pretraining means either retraining or dropping that star from the test.

## Step 3: clean each light curve

Clean only what is broken, and flag rather than delete anything a model might want to see. Flares, ramps after gaps and odd slots are part of what the search will face, so pretraining should see them; a flag lets any loader drop them later without a rebuild.

**Technical details** (per source row, streamed one parquet row group at a time; never load a whole file)

1. Sort by time; drop bins whose time or flux is not finite or whose flux is zero or negative; drop repeated timestamps (keep the first).
2. Kepler only: convert time from BKJD to BTJD (BTJD = BKJD - 2167) so all three sources share one clock.
3. FFI only: a NaN centroid or background keeps its slot; that channel is marked invalid for the slot (Step 6).
4. FFI only: check for repeated (tic\_id, sector) rows. The FFI run was restarted once, so keep the copy with more bins and log the duplicate.
5. Drop rows with fewer than 200 good bins (about 4 days); they cannot fill a meaningful window.
6. Record `quality_policy` per row: `bitmask17087` for FFI, `quality0` for 2-min and Kepler. The original flags are not in the data, so the two policies cannot be made identical; recording them lets you test whether it matters.

**Decision 3a: outliers**

| Option | What happens | Risk |
| --- | --- | --- |
| **A. Keep all, flag single-slot spikes above +8σ (recommended)** | Data untouched; a flag bit marks isolated upward spikes (cosmic rays, glints) | None; a loader can mask flagged slots if an ablation shows they hurt |
| B. Delete single-slot upward spikes above +8σ | Slightly cleaner reconstruction targets | Permanent; also clips the peak of short flares |
| C. Sigma-clip both directions | Cleanest-looking data | Deletes deep transits and eclipses, the signals you are searching for |

**Decision 3b: ramps after gaps**

| Option | What happens |
| --- | --- |
| **A. Flag the first 6 hours after any gap of 12 hours or more (recommended)** | Data kept; loaders can mask or down-weight those slots |
| B. Trim those 6 hours | Removes thermal ramps and some systematics, but also any transit that starts right after a gap |
| C. Do nothing | Simplest; the model has to learn ramps with no hint |

Recommendation for both: A. Each keeps the decision reversible at load time, which is what the proxy experiments need.

## Step 4: put every light curve on one 30-minute grid

Snap every bin from every source onto a single global grid of 30-minute slots, and store each star-sector or star-quarter as one gridded segment. Windows are not stored; a small index lists the default ones, and loaders cut their own.

&#91;embedded content: gridding and windowing · schematic, Kepler cadence offsets illustrative\]

In the top half, each bin lands in the slot that contains its time; the two Kepler cadences that fall in the same slot are averaged. In the bottom half, a TESS sector yields one 27-day window and a Kepler quarter yields three, plus an optional end-aligned fourth.

**Technical details**

1. **Grid.** Slot index k = floor((t - T0) / 30 min) with t in BTJD and one fixed epoch for every source. T0 = -2200.0 BTJD keeps every Kepler and TESS slot index positive.
2. **Snapping.** Each bin goes to the slot containing its time, so its timestamp moves by at most 15 minutes. That is small next to a 4-hour patch, but the injection and labeling code must snap times the same way.
3. **Collisions** (about one Kepler slot in 50). Combine the bins with an inverse-variance weighted mean for flux, centroids and background, combine the errors, add their point counts, and set the "merged" flag bit.
4. **Segment.** slot0 = first observed slot, n\_slots = last observed slot - slot0 + 1. Unobserved slots hold 0 with the observed bit cleared. Store n\_obs and coverage = n\_obs / n\_slots.
5. **Window index.** A separate table with one row per default window: segment\_id, start offset, length 1,296, coverage. TESS: one window at offset 0, plus an end-aligned window if 288 or more slots (6 days) remain. Kepler: windows at offsets 0, 1,296 and 2,592, plus an end-aligned window if 648 or more slots remain. Windows under 50% coverage stay listed with a flag.

**Decision 4a: the time grid**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. One global grid (recommended)** | Every star in a sector shares slot boundaries, so momentum dumps and scattered light line up across stars; cross-star features become possible later | Bins shift by up to 15 minutes |
| B. Grid per segment, starting at its first bin (what the 2-min repo did) | No shift for the first bin | Stars are not aligned with each other; Kepler slots still need re-binning |
| C. Keep exact timestamps, no grid | Most faithful | Needs a time-aware model (irregular inputs); larger storage; unlike every design so far |

**Decision 4b: what one stored row is**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Segments plus a window index (recommended)** | Any window length or offset at load time; random-offset Kepler windows for free | Loader does one crop per example |
| B. Fixed 1,296-slot windows | Loader is trivial | Window length and Kepler offsets are frozen; a different length means a rebuild |
| C. One row per star, all segments joined | Ready for multi-sector pretraining | Very uneven row sizes; A can produce this at load time by grouping on star\_key |

Recommendation: A for both. The global grid costs nothing at this cadence, and segment rows keep the window length an open experiment.

## Step 5: compute normalization statistics

Measure each segment's noise from point-to-point differences and store flux in units of that noise, keeping the statistics so the original flux can be recovered. The example star shows why: its pipeline error is 892 ppm, its point-to-point scatter is 1,190 ppm, and a plain MAD gives 4,330 ppm because it counts the star's own oscillation as noise.

**Technical details** (over observed slots of one segment)

- **flux\_median** m = median flux.
- **sigma\_p2p** = 1.4826 × MAD of differences between adjacent observed slots, divided by √2. Use only pairs of neighboring slots, not pairs across a gap.
- **sigma\_mad** = 1.4826 × MAD of (flux - m). **var\_ratio** = sigma\_mad / sigma\_p2p: about 1 for a quiet star, 3.6 for the example. Step 9 samples on it.
- **sigma\_6h** = scatter of 12-slot (6-hour) means after removing a 2-day running median. It is a rough noise level on transit timescales, which the fine-tuning injector needs for drawing S/N.
- **bkg\_sigma** (FFI only) = point-to-point scatter of the background column, the same way.
- **Centroid medians** per axis, subtracted in Step 6.
- Then flux\_z = (flux - m) / sigma\_p2p for every observed slot.

**Decision 5a: which noise scale**

| Option | Example star | Effect |
| --- | --- | --- |
| **A. Point-to-point MAD (recommended)** | 1,190 ppm | Matches the real slot-to-slot scatter; variability stays visible as large values |
| B. Plain MAD around the median | 4,330 ppm | Variable stars look quiet; a transit on them looks 3.6 times shallower than it is |
| C. Median pipeline error (flux\_err) | 892 ppm | Underestimates real scatter here; definitions differ between FFI, 2-min and Kepler |
| D. No scaling (store ppm) | none | Exact, but a 500 ppm dip means different things on different stars; the model has to learn the scaling |

**Decision 5b: over what span**

| Option | Notes |
| --- | --- |
| **A. Per segment (recommended)** | One sector or quarter; a loader can re-center and re-scale each window cheaply if a proxy experiment asks for it |
| B. Per stored window | Only possible if windows are stored (Decision 4b-B) |
| C. Per star across all sectors | Breaks when apertures, contamination or cameras change between sectors |

**Decision 5c: storage precision for flux**

| Option | Size | Precision |
| --- | --- | --- |
| **A. float16 flux\_z plus the statistics (recommended)** | 2 bytes per slot | Steps of 0.002σ at 3σ and 0.25σ at 500σ: far below the noise |
| B. float32 relative flux | 4 bytes per slot, about 17% more in total | Exact |
| C. int16 with a fixed scale (z × 100) | 2 bytes per slot | Saturates at ±327σ, so deep eclipses on quiet stars clip |

Recommendation: A for all three. Point-to-point noise is the scale a transit is judged against, and float16 loses nothing the model can use.

## Step 6: build the per-slot channels

Store seven channels per slot, 12 bytes in all, with an explicit validity bit wherever a source lacks a channel. A missing value is stored as 0 with its bit cleared, so the model can tell "no shift" from "no data".

| Channel | Definition | Type | FFI | 2-min | Kepler |
| --- | --- | --- | --- | --- | --- |
| flux\_z | (flux - median) / sigma\_p2p (Step 5) | float16 | yes | yes | yes |
| flux\_err\_z | flux\_err / sigma\_p2p | float16 | yes | yes | yes |
| cen\_x, cen\_y | (centroid - segment median) × plate scale, arcsec: 21.0″ per pixel for TESS, 3.98″ for Kepler | float16 each | NaN allowed, flagged | yes | yes |
| bkg\_z | (bkg - median bkg) / sigma\_p2p. bkg is already in units of the star's flux, so a background bump is in the same σ units as a flux dip | float16 | yes | 0, flagged | 0, flagged |
| fill | Share of the expected raw points in the slot, 0-255 | uint8 | n\_points / 1, 3 or 9 (sectors 1-26, 27-55, 56+) | estimated: (median flux\_err / flux\_err)², capped at 1 | 1 |
| flags | bit 0 observed, 1 centroid valid, 2 background valid, 3 fill estimated, 4 merged, 5 spike above +8σ, 6 within 6 h after a gap of 12 h or more, 7 spare | uint8 |  |  |  |

**Technical details**

- Write each channel as one little-endian byte array per row (n\_slots × 2 bytes for float16, n\_slots × 1 for uint8). Any language can decode it with one call, such as numpy.frombuffer.
- Centroids are detector axes. Kepler rotates 90° every quarter and TESS cameras differ in orientation, so x and y do not mean the same sky direction across segments. The size of a shift is consistent; its direction is not.
- The 2-min fill estimate works because each 30-minute bin averages up to 15 two-minute points, and its error grows as fewer points survive. Mark it with bit 3 so a loader can ignore it.

**Decision 6a: which channels to store**

| Option | Size | What you can test later |
| --- | --- | --- |
| **A. All seven (recommended)** | 12 bytes per slot | Every channel ablation from the plan: centroid, background, fill, error weighting |
| B. Flux and flags only | 3 bytes per slot, about 4× smaller | Nothing about centroids or background without a rebuild |
| C. Seven plus derived channels (detrended flux, centroid magnitude) | 16+ bytes per slot | Nothing new: these are cheap to compute at load time |

**Decision 6b: centroid form**

| Option | Notes |
| --- | --- |
| **A. x and y in arcsec (recommended)** | Same units across missions; a loader can turn them into a magnitude r = √(x² + y²) |
| B. Magnitude r only (what Salinas et al. use) | Orientation-free, but loses the sign and direction of a shift for good |
| C. Pixels | Kepler pixels are about 5× smaller than TESS pixels, so the same shift means different things |

Recommendation: A for both. Storing x and y leaves r as a load-time choice, and dropping channels later is free while adding them is a rebuild.

## Step 7: attach metadata and tags

Give every row the metadata a loader needs to filter, weight and trace it, plus tags from the catalogs. Tags describe a row; they never remove it. Removal happened once, in Step 2. Tags are never model inputs or targets: pretraining stays label-free, and the tags exist only for exclusion, sampling, ablations, fine-tuning and evaluation. Keep them out of the metadata token.

| Group | Columns (type) |
| --- | --- |
| Identity | segment\_id (string, such as `ffi:s0001:29257288` or `kep:q07:757076`), star\_key, tic\_id and kic\_id (int64, null when absent), source (ffi, 2min, kepler), sector or quarter (int16), cycle (int8, TESS only), native\_cadence\_s (1800, 600, 200, 120 or 1765) |
| Grid | slot0 (int32), n\_slots, n\_obs (int16), coverage (float32) |
| Normalization | flux\_median, sigma\_p2p\_ppm, sigma\_mad\_ppm, var\_ratio, sigma\_6h\_ppm, bkg\_sigma, cen\_x\_median\_px, cen\_y\_median\_px (float32) |
| Star | tmag, kepmag, teff, radius (float32, null when missing) |
| Catalog tags | toi (TOI numbers, joined), toi\_disposition, toi\_period\_d, kepler\_tce\_labels (DR24 PC, AFP, NTP), eb\_catalog (tess\_eb, kepler\_eb or none) |
| Known events | list of (center slot, duration in slots, kind) predicted inside this segment from TOI, KOI and EB ephemerides |
| Bookkeeping | dup\_group, quality\_tier, quality\_policy, hash\_bucket, split |
| Provenance | source\_repo, source\_file, source\_row\_group, source\_row, build\_version, build\_commit |

**Technical details**

- Join catalogs on star\_key after the KIC-to-TIC crossmatch, so a TOI on a Kepler-field star also tags its Kepler rows.
- **Known events**: for each TOI, KOI or EB with an ephemeris, compute every predicted mid-time inside \[slot0, slot0 + n\_slots), snapped with the Step 4 grid. Single-transit TOIs with no period get their one epoch if it falls inside.
- Keep the per-row provenance exact (file, row group, row). Any odd row in training can then be traced to its source in one lookup.

**Decision 7: how much catalog knowledge to bake in**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Tags plus known events (recommended)** | Enables, with no rebuild: masking known transits during pretraining (the EXOVEIL-style ablation), drawing real-transit windows for fine-tuning, and checking what the model does on known planets | Needs the ephemeris export from Step 0 |
| B. Star-level tags only | Simpler join | Event-level uses need a second pass later |
| C. No tags; join catalogs at load time | Nothing to keep updated | Every loader repeats the join; catalogs change, so two teammates can get different answers |

Recommendation: A. Catalogs change over time, so record the catalog download dates in the dataset card and rebuild the tags (metadata only) when you refresh them.

## Step 8: handle duplicates and grade quality

Keep both reductions when a star-sector exists in 2-min and FFI form, and grade every row into a quality tier instead of deleting it. Only rows too noisy to hold any detectable transit are dropped.

**Technical details**

- **dup\_group** = hash of (tic\_id, sector), shared by a star-sector's FFI and 2-min rows. Step 11 reports how many pairs exist; most 2-min targets are bright enough to be in the FFI list too.
- **quality\_tier**, starting thresholds (check them against the Step 11 histograms before freezing):

| Tier | Rule | Meant for |
| --- | --- | --- |
| 1 clean | coverage ≥ 0.8 and sigma\_p2p ≤ 2,000 ppm | All uses |
| 2 usable | coverage ≥ 0.5 and sigma\_p2p ≤ 10,000 ppm | All uses |
| 3 poor | everything else that survives | Pretraining only, down-weighted |
| dropped | sigma\_p2p > 50,000 ppm (5% scatter) | Nothing. The handoff doc notes light curves with about 17% scatter, where no planet signal could be seen |

**Decision 8a: FFI and 2-min duplicates**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Keep both, linked by dup\_group (recommended)** | Two independent reductions of the same photons are free, realistic augmentation: the model learns that FFI noise and 2-min noise belong to the same star | The sampler must not count a pair as two stars (Step 9 counts both toward the star's cap) |
| B. FFI only | Everything looks like the search data | Discards 2-min's cleaner light curves |
| C. 2-min only for those stars | Cleanest version | FFI is what you will search; pretraining should see FFI for bright stars too |

**Decision 8b: what to drop for quality**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Tiers, drop only above 5% scatter (recommended)** | Every filtering choice stays a load-time switch | Larger dataset |
| B. Also drop tier 3 at build time | Smaller, cleaner | Reversing it is a rebuild; the search will still meet poor light curves |
| C. Drop nothing | Complete | Keeps rows with no usable signal |

Recommendation: A for both.

## Step 9: store default sampling weights

Store one default weight per window, and its parts, so every teammate's loader draws the same mix unless an experiment says otherwise. Without weights, a uniform draw would be 55% FFI, 37% Kepler and 8% 2-min by window count, and stars with 30+ sectors would dominate.

**Technical details** (computed on the window index, after Steps 4 and 8)

| Part | Rule | Default |
| --- | --- | --- |
| w\_source | target share of the source / number of windows from that source | FFI 50%, 2-min 25%, Kepler 25% |
| w\_star | min(1, cap / windows this star has across all sources, duplicates included) | cap = 20 |
| w\_var | 2 for windows in the top 20% of var\_ratio within their source, else 1 | Variable stars and messy systematics are what pretraining should learn most from |
| w\_tier | 1 for tiers 1-2, 0.5 for tier 3 |  |
| w\_default | product of the four, rescaled to sum to the number of windows |  |

The 55% / 37% / 8% split assumes about 12.8M FFI windows, 8.5M Kepler windows once q15-q17 land, and 1.7M 2-min windows.

**Decision 9: weights in the data or in the loader**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Store the parts and a default product (recommended)** | Everyone gets the same mix by default; an experiment changes one part, such as w\_source for the FFI-only proxy run | Weights must be recomputed if the source mix target changes (metadata only, minutes) |
| B. Uniform sampling | Nothing to compute | Mix follows window counts, not the plan; long-observed stars dominate |
| C. No stored weights; each loader computes its own | Most flexible | Two runs that should differ only in model can differ in data too |

Recommendation: A. The source shares are the first proxy experiment, so treat 50/25/25 as a starting point.

## Step 10: write, shuffle and upload

Run the build in two passes: a parallel, resumable pass per source file, then one global pass that grades, weights, shuffles and shards. Write Parquet with each channel as a raw byte column, pre-shuffled, into one public pretraining repo and one private held-out repo.

**Technical details**

1. **Pass 1, one job per source file** (FFI sector, 2-min sector, Kepler quarter; 206 jobs in all). Stream row groups; run Steps 3-7; route each row to pretrain or heldout by Step 2; append it to one of 256 staging buckets chosen by a seeded random number. Write a done-marker per source file so a killed job resumes where it stopped.
2. **Pass 2, global.** Read the metadata columns only (small) to compute quality tiers, var\_ratio quantiles, per-star window counts and weights (Steps 8-9). Then, bucket by bucket, shuffle the rows in memory and write final shards of about 1 GB. Bucketing in pass 1 and shuffling inside buckets gives a full shuffle without ever holding the dataset at once.
3. **Parquet settings:** zstd compression, row groups of about 64 MB, channels as binary columns of little-endian bytes, metadata as typed columns.
4. **Layout:**
   - `nyu-planet-finders/lc-pretrain` (public): configs `ffi`, `2min` and `kepler`, each with train and validation splits; config `index` (windows.parquet with weights, stars.parquet); config `proxy` (a stratified 200k-segment sample of train plus 10k of validation, about 4 GB, for Kaggle).
   - `nyu-planet-finders/lc-heldout` (private): the same layout for search-holdout, scoreboard hosts and list exclusions, plus exclusions.parquet. At roughly 10-15% of the build it stays under the 100 GB private limit.
5. **Dataset card:** build version, git commit, every decision taken in this guide, catalog download dates, column definitions, the TESS-SPOC and MAST credits (Caldwell et al. 2020, DOI 10.17909/t9-wpz1-8s54).

**Decision 10a: file format**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Parquet, channels as byte columns (recommended)** | Smallest; column filtering (read metadata without arrays); any language decodes it | The Hub's web viewer shows bytes, not values |
| B. Parquet, channels as list of float32 | Viewer shows numbers; no decoding step | About 2× larger; float32 brings no real precision gain here |
| C. WebDataset tar of .npy files | Fastest sequential streaming | No column filtering; metadata lives in side files |
| D. Hugging Face Arrow (save\_to\_disk) | Loads directly with the datasets library | Large, uncompressed, tied to one library |

**Decision 10b: where to run it**

| Option | Fit |
| --- | --- |
| **A. Torch HPC CPU nodes (recommended once you have access)** | Many cores and disk; pass 1 runs 206 jobs in parallel |
| B. TIKE | Proven for streaming from S3 and the Hub, but 2 cores and 7.6 GB RAM make it slow; staging must be uploaded and deleted as you go |
| C. Kaggle CPU notebooks | 4 cores, 29 GB RAM; about 20 GB of working disk and session limits force one source file per session |
| D. A rented CPU machine | Fast to start, costs a little; pick one with \~1 TB of disk for staging |

Recommendation: A for format and A for compute. If Torch access is still pending, D is the fastest unblock; B works if nothing else does.

The 206 jobs are 81 FFI sectors, 107 2-min sectors and 18 Kepler quarters.

## Step 11: check the build before anyone trains on it

Run these checks on the finished build and tick them off. The first three are hard gates: if any fails, fix and rebuild before training. The rest catch subtler bugs that would otherwise surface weeks later as a strange proxy result.

- [ ] **Row accounting.** For each source file, rows in = rows written + rows dropped, with a reason for every drop (non-finite, under 200 bins, over 5% scatter, FFI duplicate).
- [ ] **Leakage audit.** Zero star\_key overlap between lc-pretrain and lc-heldout, checked after the KIC-to-TIC mapping and across all three sources.
- [ ] **Round trip.** For 1,000 random segments, decode flux\_z × sigma\_p2p + median and compare with the source flux at the same times. Differences should stay under 0.1σ (float16 rounding plus snapping).
- [ ] **Unread columns.** Check FFI cen\_x, cen\_y, bkg and n\_points against expectations: centroid offsets centered near 0, n\_points of about 1, 3 and 9 changing at sectors 27 and 56. I could not read these columns from this session, so this is their first real check.
- [ ] **Grid alignment.** For one sector, take the per-slot median of flux\_z over 1,000 FFI stars. Shared features such as momentum dumps should appear as sharp common features at fixed slots.
- [ ] **Noise versus magnitude.** Plot sigma\_p2p against Tmag per source. Real data follows a smooth noise floor; a cloud of outliers usually means a normalization bug. Set the Step 8 tier thresholds from this plot.
- [ ] **Duplicates.** Count dup\_group pairs; for 100 pairs, the FFI and 2-min flux\_z should correlate strongly.
- [ ] **Known events.** For 50 confirmed planets with good ephemerides, plot a day around each known event; the dip should sit within one slot of the predicted center.
- [ ] **Eyeball pass.** Two people look at 50 random segments per source and tier.
- [ ] **Loader smoke test.** A reference loader (crop a window, compress flux, patch) reads the proxy config on Kaggle fast enough to keep a T4 busy.
- [ ] **Release.** Dataset card complete, build tagged v1.0, handoff doc updated.

## The datasets your decisions produce

Build once, with the recommended choices, as **lc-pretrain v1**. The named pretraining datasets are then just which configs a loader reads, and every experimental knob lives in the loader. A new build is needed only when a left-column decision changes.

&#91;embedded content: decision map · sizes estimated at 12 bytes per slot\]

| Variant | Configs read | Segments | Default windows | Size before compression | Use it for |
| --- | --- | --- | --- | --- | --- |
| **P-ALL (recommended)** | ffi + 2min + kepler | about 17.4M | about 23M | about 370 GB | The main pretraining run |
| P-TESS | ffi + 2min | about 14.5M | about 14.5M | about 233 GB | Testing whether Kepler helps |
| P-FFI | ffi | about 12.8M | about 12.8M | about 205 GB | The target-domain-only baseline |
| Proxy | proxy | 200k | about 260k | about 4 GB | Every Kaggle proxy experiment |

Counts assume Kepler q15-q17 are finished; without them P-ALL has about 2.4M Kepler segments instead of 2.9M. At 163 tokens per window, P-ALL is about 3.8 billion tokens per pass, far more than a 5M-parameter model needs in one pass. The per-star cap and the weights decide what is actually seen.

**Order of work:** Step 0, then Steps 1-2 (freeze the exclusion list with the team), then a pass-1 trial on one FFI sector, one 2-min sector and one Kepler quarter, then Step 11's hard gates on that trial, and only then the full build. The trial costs a few hours and catches most bugs before 370 GB are written.

## Decision summary and the default loader

Every decision in this guide, with the recommended choice. Baked decisions need team agreement before the build; load-time ones are proxy experiments.

| # | Decision | Recommended | Main alternative | When it is fixed |
| --- | --- | --- | --- | --- |
| 0 | Kepler q15-q17 | Finish them first | Build now, add as v1.1 | Before build |
| 1 | Splits | Hash buckets on star\_key, 85/5/5/5 | Stratified by Tmag | Baked |
| 2 | Exclusions | Scoreboard stars only; long-period cut at 20 days | Also every TOI and EB star | Baked (B is a load-time filter) |
| 3a | Outliers | Keep, flag spikes above +8σ | Delete spikes | Flag baked, use at load time |
| 3b | Post-gap ramps | Keep, flag 6 h | Trim 6 h | Flag baked, use at load time |
| 4a | Time grid | One global 30-minute grid | Grid per segment | Baked |
| 4b | Stored row | Segment plus window index | Fixed 1,296-slot windows | Baked |
| 5a | Noise scale | Point-to-point MAD | Plain MAD | Baked |
| 5b | Normalization span | Per segment | Per window | Baked; re-normalizing per window is a load-time option |
| 5c | Flux precision | float16 z plus statistics | float32 relative flux | Baked |
| 6a | Channels | All seven | Flux and flags only | Baked |
| 6b | Centroid form | x, y in arcsec | Magnitude r only | Baked; r is a load-time option |
| 7 | Catalog knowledge | Tags plus known events | Star-level tags only | Metadata; refresh without a rebuild |
| 8a | FFI / 2-min duplicates | Keep both, linked | FFI only | Load-time filter |
| 8b | Quality drops | Tiers; drop only above 5% scatter | Also drop tier 3 | Baked drop, tiers at load time |
| 9 | Sampling | Stored weights, 50/25/25 | Uniform | Load time |
| 10a | Format | Parquet, byte columns | Parquet, float32 lists | Baked |
| 10b | Build compute | Torch CPU nodes | Rented CPU machine | Before build |

### What a default loader does with P-ALL

These are the starting settings for Stage 0 of the proxy plan; each is one config line to change.

1. Draw a window from windows.parquet with probability w\_default; for Kepler and long TESS segments, move its start by a random offset of up to the spare slots.
2. Decode the seven channels; mask slots whose observed bit is 0.
3. Compress flux\_z with asinh beyond ±10σ instead of a hard clip.
4. For half the Kepler windows, add Gaussian noise so their sigma matches an FFI star of a randomly drawn magnitude, and blank a 1-day mid-window gap.
5. Blank one or two extra random spans of 0.5-2 days in 20% of windows.
6. Hide about 35% of patches in runs of 1-6 patches (4-24 hours), hiding only flux in 25% of batches.
7. Hand the model the slot channels plus a metadata token: log10 sigma\_p2p, magnitude, Teff, log radius, source, native cadence and missing flags.
