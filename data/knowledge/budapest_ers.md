# Budapest ERS Deployment
The Hungaroring has a single DRS zone and a low-energy deployment profile due to its slow overall pace. The primary strategic value of ERS here is defending position on the back straight rather than pure lap time.

Key deployment zones:
- Back straight (Turn 14 to Turn 1): Single DRS zone; full deployment; reaches ~290 km/h. Adds approximately 0.25 s.

Key harvest zones:
- Turn 1 braking: Primary harvest; deceleration from 290 km/h into medium-speed corner.
- Turn 4 hairpin braking: Hard stop; good harvest event.
- Turn 11 braking: Medium harvest in the technical middle sector.

Battery management thresholds:
- Single DRS zone means the lowest per-lap energy demand of any circuit.
- Nominal SOC entering back straight: 0.35+.
- SOC below 0.25: reduce approach deployment before Turn 14; maintain full DRS straight deployment.
- SOC below 0.15: full harvest at Turns 1 and 4; minimal deployment.
- SOC below 0.12: full harvest mode. Alert engineer.

Defensive note: With overtaking so difficult, being beaten by a rival at Turn 1 after the DRS zone is nearly impossible to reverse. Ensure SOC is ≥0.45 for the lap when defending from a faster car in a critical race window. ERS here is primarily tactical, not pace-setting.
