# Data Generation — Assumptions and Provenance

This document exists so every number and rule in `harbour_line_ridership.parquet`
can be traced back to either (a) a real, cited fact, or (b) an explicit,
justified modeling choice. Nothing in the generator is an unexplained magic
constant — search this file for the constant if you're unsure where it came
from.

The dataset is **synthetic**: no real Mumbai Suburban Railway ridership or
occupancy data exists publicly at this granularity, so it could not have
been collected. It is built from (1) real, verified geography and climate
facts and (2) an explicit, documented simulation of commuter behavior on
top of those facts. Phase 5 (the YOLO demo) is framed explicitly as "how
the occupancy target would actually be measured/labeled in a real
deployment" — this document is the specification that a real data-collection
and labeling pipeline would be validated against.

Regenerate the exact dataset committed to this repo with:
```
python -m src.data_generation.generate_dataset
```
It is fully deterministic given `--seed` (default 42).

---

## 1. Station geography — `src/data_generation/stations.py`

**Real, verified facts:**
- The Harbour Line's classic CSMT↔Panvel trunk has 25 stations, in the
  order encoded in `stations.py`: CSMT, Masjid, Sandhurst Road, Dockyard
  Road, Reay Road, Cotton Green, Sewri, Wadala Road, GTB Nagar,
  Chunabhatti, Kurla, Tilak Nagar, Chembur, Govandi, Mankhurd, Vashi,
  Sanpada, Juinagar, Nerul, Seawoods-Darave, Belapur CBD, Kharghar,
  Mansarovar, Khandeshwar, Panvel.
- This was verified via live web search cross-referencing Wikipedia's
  "Harbour line (Mumbai Suburban Railway)" article summary, a NAVITIME
  transit stop-list for the CSMT–Panvel route, and a MumbaiLocal.Info
  route guide, on 2026-09-03. **Direct fetch of en.wikipedia.org was
  blocked by this sandbox's network egress policy** (attempted and
  logged — see §5), so the list is reconstructed from independent
  secondary-source summaries of that page rather than the page itself.
- **Correction to the project brief:** the brief's example station list
  included "Vidyavihar" and "King's Circle". Verification found Vidyavihar
  is on the Central Line (between Kurla and Ghatkopar), not the Harbour
  Line — excluded. King's Circle is on the CSMT–Goregaon branch (via
  Wadala→Mahim), not the CSMT–Panvel trunk this project models — also
  excluded. Both are called out here explicitly per the brief's own
  request for examiner-defensibility.
- Real-world scope simplification: the full Harbour Line network has ~35
  stations across three branches (CSMT–Panvel, CSMT–Goregaon, and the
  Trans-Harbour Vashi–Thane line). This project models only the
  CSMT↔Panvel trunk — the original and highest-ridership branch — as a
  single linear corridor with no branching.
- Interchange stations (`is_interchange=True`): CSMT (Central Line), Wadala
  Road (Monorail + branch junction), Kurla (Central Line), Vashi (major
  Navi Mumbai bus/road hub), Panvel (Central Line + long-distance/Konkan
  Railway).

**Modeling choices:**
- `distance_km` is an approximate cumulative distance from CSMT, assembled
  from the well-documented ~54 km trunk length with individual
  inter-station spacing interpolated proportionally — **not** survey-grade
  GPS data. It is used only for *relative* modeling (how full a train has
  become by the time it reaches a station), never presented as
  authoritative.
- `station_type` (`cbd_terminal`, `industrial_dock`, `interchange_hub`,
  `secondary_cbd`, `residential`, `terminal_hub`) is an original
  classification for this project, based on each station's real character
  (old Bombay Port Trust dockland for Sandhurst Road→Sewri; Navi Mumbai's
  planned CBD at Belapur; major interchanges at Wadala/Kurla/Vashi/Panvel;
  everything else residential suburb). It exists specifically to give
  Phase 2's clustering genuine, defensible shape diversity to *discover* —
  see §3.

---

## 2. Weather — `src/data_generation/weather.py`

**What was attempted:** the brief asked to pull real historical weather
where feasible. Two live attempts were made and both were blocked by this
sandbox's network egress policy (not a data-availability problem — both
sources are normally public/free):
1. `archive-api.open-meteo.com` (Open-Meteo Historical Weather Archive,
   no API key required, hourly/daily Mumbai coverage back to 1940) —
   `curl` returned a proxy-level 403 policy denial on the CONNECT tunnel.
2. `en.wikipedia.org/wiki/Climate_of_Mumbai` (for a full tabulated monthly
   normals table) — also blocked by egress policy.

**Fallback per the brief:** daily weather is generated synthetically, but
*calibrated* to real published climate normals obtained via live web
search (which routes through a different, unblocked path), not invented:
- Annual rainfall (~2,502.3 mm) and per-month rainfall normals for the
  India Meteorological Department's Santacruz observatory (Mumbai's
  official long-period-average reference station) were corroborated
  directly from search results citing IMD data. See
  `MONTHLY_RAINFALL_MM` in `weather.py`.
- Monthly mean max/min temperatures were corroborated from search results
  ("mean maximum ~32°C in summer / ~30°C in winter, mean minimum ~26°C
  summer / ~18°C winter") and interpolated month-to-month into
  `MONTHLY_TEMP_MAX_C` / `MONTHLY_TEMP_MIN_C`.
- IMD's standard 24-hour rainfall intensity classification (light
  2.5–15.5 mm, moderate 15.6–64.4 mm, heavy 64.5–115.5 mm, very heavy
  115.6–204.4 mm, extremely heavy >204.4 mm) is IMD's published
  classification, used directly (`_rainfall_intensity_band`) to decide
  monsoon-disruption regimes in §3.

**Generation method:**
- Each day's wet/dry state is drawn from a 2-state Markov chain with
  month-dependent persistence (`MONTHLY_WET_PERSISTENCE`), so monsoon rain
  arrives in realistic multi-day bursts rather than independent daily
  coin-flips — matching how real monsoon spells behave.
- Wet-day rainfall amounts are drawn from a Gamma distribution
  (shape=1.3, giving a heavier right tail than an exponential — occasional
  deluge days) whose mean is tuned so the *expected* monthly total matches
  the calibrated normal for that month. Because this is one stochastic
  realization of a full year, actual monthly totals land close to but not
  exactly on the targets (see the validation table in the Phase 1
  check-in) — this mirrors how real years vary around their long-period
  average, not a calibration bug.
- Temperature follows a smooth seasonal curve through the monthly normals
  (via day-of-year interpolation) with AR(1) day-to-day noise, and is
  reduced further on rainy days — Mumbai measurably cools during heavy
  rain, a real and well-documented effect.

---

## 3. Ridership simulation — `src/data_generation/simulate.py`

This is the core, most consequential set of modeling choices. `occupancy_pct`
is expressed as **% of a coach's rated carrying capacity**, and is
deliberately allowed to exceed 100%.

**Real, cited fact driving the value range:** Mumbai's suburban trains are
widely reported to run at "super-dense crush loads" of **2.5×–3× their
rated capacity** during peak hours (e.g. a 12-car rake rated for ~1,700–2,400
passengers regularly carries 4,500–6,000 at peak; commonly cited peak
figures include ~16 standing passengers per square metre of floor). This is
why `occupancy_pct` is clipped to **[0, 300]**, not [0, 100] — a value of
100% means "at rated capacity", not "the maximum possible". Sourced via
live web search (2026-09-03), corroborating multiple independent news/
analysis sources on Mumbai suburban rail crowding statistics.

**The core formula** (see the module docstring in `simulate.py` for the
full narrative) sums two independently-scaled terms:

- **`floor`** — a constant off-peak baseline present at all hours.
- **`peak_swing`** — the AM/PM commute bulge, modeled as two Gaussian bumps
  in hour-of-day (μ=8.5, μ=19.0), whose *amplitude at a given station*
  depends on how far the train has "filled up" by the time it reaches that
  station:
  - **UP** (CBD-bound, towards CSMT): amplitude grows with proximity to
    CSMT — the train accumulates boarders the whole way from Panvel, so
    it's most crowded just before CSMT and emptiest near the Panvel end.
  - **DOWN** (towards Panvel): the mirror image — amplitude grows with
    distance *from* CSMT, peaking near Panvel in the evening.
  - This produces the monotonic-with-distance buildup validated in the
    Phase 1 check-in (AM peak, UP, General coach: ~52% at Panvel rising to
    ~200% at CSMT on an ordinary Monday).
  - A smaller reverse-direction component (0.25–0.35× the main peak
    amplitude) represents realistic minor reverse-commute traffic.

Both terms are then scaled, in order, by:
1. **Day-type multiplier** (`peak_day_mult` / `floor_day_mult`): weekday
   1.0 → Saturday 0.80–0.90 → Sunday 0.55–0.75 → holiday 0.45–0.70 (peaks
   flatten much more than the off-peak floor does, since holidays still
   carry some leisure travel).
2. **Mega-block multiplier**: Indian Railways runs planned Sunday
   maintenance "mega blocks" that suspend/reduce service on one line for a
   several-hour window. This project simplifies that to a whole-day
   per-date flag (`is_mega_block_day`, ~40% of Sundays, drawn in
   `calendar_features.py`) whose crowding effect (×1.6, fewer trains
   carrying the same demand) is applied only during a simulated
   **11:00–16:00** block window — approximating real practice (blocks are
   hours, not the whole day) while keeping the calendar feature itself a
   simple boolean. Documented simplification.
3. **Monsoon rain-regime multiplier** (`compute_monsoon_regime`): one
   regime is drawn *per date* (not per row, so a day's disruption is
   coherent across every station/hour that day), keyed off the IMD
   intensity band:
   - `moderate` rain (15.6–64.4 mm): deterministic small crowding
     increase (commuters rush to beat disruption, avoid flood-prone road
     alternatives) — up to +15% at peak.
   - `heavy` rain (64.5–115.5 mm): 70% chance **"strained"** (fewer trains
     running, ×1.3–1.6 crowding on survivors) vs. 30% chance
     **"suspended"** (line paused, ×0.3–0.5).
   - `very_heavy` / `extremely_heavy` (>115.5 mm): 85% chance
     **"suspended"** (×0.15–0.4) vs. 15% **"last trains packed before
     suspension"** (×1.4–1.8) — modeling both regimes the brief explicitly
     asked for (crowding *increases* from disruption pile-up AND
     *decreases* from suspension), stochastically rather than picking one.
   - This mirrors real, well-known event categories on Mumbai's suburban
     network (the 2005, 2017, 2019 and 2021 deluge-driven suspensions are
     the real-world analogue), without claiming to model any specific
     historical event.
4. **Station-type shape multiplier** (`_station_shape_multiplier`) — this
   is the deliberate design choice that gives Phase 2's clustering
   something real to find:
   - `industrial_dock` (Sandhurst Road…Sewri): ×0.55 at peak, ×0.35
     off-peak — low local trip generation (historically low residential
     density in the old dockland), so these stations contribute little
     beyond pass-through peak traffic.
   - `interchange_hub` (Wadala Road, Kurla, Vashi): ×1.0 at peak (unchanged
     — these legitimately see the full route-level peak) but ×1.35
     off-peak — constant interchange foot traffic keeps midday/evening
     ridership elevated, unlike a quiet residential stop.
   - `secondary_cbd` (Belapur CBD): ×1.0/×1.1, **plus** an additive
     reverse-commute bump — Navi Mumbai's own business district draws
     workers from the CSMT-side of the corridor, so it gets its own local
     AM-in (DOWN direction) / PM-out (UP direction) pattern layered on top
     of the route-level commute pattern. This assumes Belapur's workforce
     predominantly commutes from the denser CSMT/Vashi side rather than
     the newer far-Panvel-side developments — a documented simplification.
   - `cbd_terminal` (CSMT) / `terminal_hub` (Panvel): ×1.0/×1.15 — modest
     all-day floor boost for terminal/interchange traffic beyond pure
     commute demand.
   - `residential` (everything else): ×1.0, unmodified — this is the
     reference "sharp peak, quiet midday and night" shape.

   **Framing note:** `occupancy_pct` is modeled as a *station-observed*
   metric (how crowded do coaches appear when a train calls at this
   station) that is intentionally allowed to vary with local station
   character, not purely with route position. This is analogous to real
   transit ridership studies, where boarding/alighting activity — not just
   onboard load — differentiates a station's role. It is a deliberate
   modeling choice made specifically so Phase 2's station clustering has
   genuine, explainable shape diversity to discover, not an attempt to
   simulate train physics.

5. **Coach-type multiplier**: `General` = 1.0 (the base value computed
   above *is* the General-coach value). `Ladies` = 1.05 − 0.20×(peak
   intensity) — relatively more used off-peak proportionally (fewer
   ladies coaches per rake, so a given coach stays more consistently
   occupied), somewhat less extreme than General at the very peak.
   `First Class` = 0.35 + 0.30×(peak intensity) — steady moderate use by
   fare-paying business commuters off-peak, still fills up but less
   extremely than General at peak (higher fare deters volume; validated in
   the Phase 1 check-in at ~64–74% of General's peak value). These ratios
   are original assumptions, not sourced figures — the direction (First
   Class < Ladies < General at peak) is the defensible part, not the exact
   coefficients.

**Noise:** a per-(station, date) random effect (N(0, 8 pct-points), one
draw shared across all hours/directions/coaches at that station on that
day — simulating a day-specific event like a signal failure or local
disruption) plus independent per-row noise (N(0, 4 pct-points)).

**Final value:** `clip(commute_component × coach_mult × 100 + noise, 0, 300)`.

---

## 4. Calendar features — `src/data_generation/calendar_features.py`

- `HOLIDAYS_2023` is a fixed, named list of Indian national + Maharashtra
  state public holidays observed in Mumbai in calendar year 2023, from
  general public knowledge of the Indian holiday calendar. It is a fixed
  date list rather than an algorithmic calculator because several holidays
  (Holi, Eid, Ganesh Chaturthi, Diwali) follow lunar/regional calendars
  that are impractical to derive with a simple rule. Documented
  simplification.
- `is_mega_block_day`: see §3 point 2 above.

---

## 5. What was blocked, for the record

This sandbox's network egress policy blocked, on 2026-09-03:
- `archive-api.open-meteo.com` (historical weather API — 403 policy denial
  on the HTTPS CONNECT tunnel).
- `en.wikipedia.org` and `mumbai.fandom.com` (via the WebFetch tool —
  `EGRESS_BLOCKED` errors).

Per this environment's own operating guidance, these denials were **not**
retried or routed around — WebSearch (which routes differently) was used
instead to corroborate the same facts from independent secondary sources,
and every fact obtained that way is flagged as such above rather than
presented as a direct primary-source pull.

---

## 6. Output

- `data/processed/harbour_line_ridership.parquet` — the full dataset,
  2023-01-01 to 2023-12-31, ~1.04M rows (365 days × 25 stations × 19
  service hours [05:00–23:00] × 2 directions × 3 coach types), ~1.8 MB.
- `data/processed/harbour_line_ridership_sample.csv` — a fixed random
  20,000-row sample (seeded) of the same data, for quick inspection
  without a parquet reader.

Regenerate with `python -m src.data_generation.generate_dataset` (or
`make data`). `--quick` restricts to one month for fast iteration.
