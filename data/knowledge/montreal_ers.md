# Montreal ERS Deployment
Montreal is one of the highest ERS-value circuits due to its two very long straights and multiple heavy braking zones that recover energy efficiently.

Key deployment zones:
- Back straight (Casino Hairpin exit to Turn 11): Primary zone; full deployment; reaches ~320 km/h. Adds approximately 0.5 s.
- Start/finish straight (Turn 13 to Turn 1): Secondary zone; full deployment; adds approximately 0.35 s.

Key harvest zones:
- Turn 10 (Casino Hairpin): Best single harvest event on the circuit; hard deceleration from 300+ km/h into the slowest corner. Very high energy recovery.
- Turn 11 (first chicane): Immediate hard braking after back straight; excellent harvest.
- Turns 13–14 (Wall of Champions chicane): Another hard braking event; strong harvest.

Battery management thresholds:
- Two long straights mean high per-lap energy expenditure; SOC management is important.
- Nominal SOC entering back straight: 0.50+. The two-straight structure requires more stored energy than single-DRS circuits.
- SOC below 0.35: reduce start/finish straight deployment; protect back straight.
- SOC below 0.22: back straight only; full harvest at braking zones.
- SOC below 0.12: full harvest mode. Alert engineer.

Safety car restart: Montreal safety cars frequently produce restarts where ERS advantage in the first two laps is decisive for track position. Ensure SOC ≥0.55 at any predicted restart lap.
