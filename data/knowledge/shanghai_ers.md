# Shanghai ERS Deployment
Shanghai has two significant DRS zones and a mixed deployment profile due to the varied speed characteristics per sector.

Key deployment zones:
- Back straight (Turn 13 exit to Turn 14): Primary DRS zone; full deployment; reaches ~315 km/h. Adds approximately 0.35 s.
- Start/finish straight (Turn 16 exit to Turn 1): Secondary DRS zone; full deployment from pit lane exit. Adds approximately 0.25 s.

Key harvest zones:
- Turn 1–2 hairpin complex: Hard braking event from 290 km/h; very high harvest.
- Turn 6 braking: Strong harvest event.
- Turn 14 braking: Hard stop from 310+ km/h; excellent harvest before the stadium section.

Battery management thresholds:
- Nominal SOC entering back straight: 0.40+.
- SOC below 0.30: reduce deployment on start/finish straight; protect back straight for overtaking potential.
- SOC below 0.20: back straight only; switch to harvest priority elsewhere.
- SOC below 0.12: full harvest mode. Alert engineer.

Spring racing note: In cooler April conditions, battery thermal management is less restrictive than summer circuits. The system may sustain higher deployment rates without thermal intervention.
