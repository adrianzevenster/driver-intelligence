# Jeddah ERS Deployment
Jeddah has the highest straight-line top speed of any street circuit, making ERS deployment extremely valuable on the long main straight DRS zone.

Key deployment zones:
- Main straight (Turn 27 chicane to Turn 1): Full deployment. The 1.9 km DRS zone is the dominant ERS consumption event each lap. Adds approximately 0.5 s per lap at full deployment.
- Sector 2 traction exits: Medium deployment from Turn 6, Turn 9, Turn 13 traction zones; adds partial lap time benefit.

Key harvest zones:
- Turn 1 braking: High-speed entry from 320+ km/h; excellent harvest event.
- Sector 2 hairpins (Turn 4, Turn 6, Turn 9, Turn 13): Each hairpin is a harvest point; the tight sector recovers a meaningful fraction of the main straight expenditure.

Battery management thresholds:
- Nominal SOC entering main straight: 0.45+. The long DRS zone requires more stored energy than most circuits.
- SOC below 0.35: reduce deployment by 30% on sector 2 traction zones; protect main straight.
- SOC below 0.25: main straight deployment only; disable sector 2 assists.
- SOC below 0.12: full harvest mode. Alert engineer immediately.

Safety car context: Under safety car conditions, ERS recharges faster than usual (no high-speed deployment drains). Re-set deployment strategy to full-attack on restart lap.
