# Lusail ERS Deployment
Lusail has a single DRS zone on the main straight, but the high-speed continuous corners mean ERS harvest events are less sharp and more spread across the lap compared to stop-start circuits.

Key deployment zones:
- Main straight (Turn 16 to Turn 1): Single DRS zone; full deployment from fast Turn 16 exit; reaches ~310 km/h. Adds approximately 0.3 s.

Key harvest zones:
- Turn 14 braking: Primary harvest; hardest single braking event; deceleration from 290+ km/h.
- Turns 7–8 deceleration: Moderate harvest from the mid-sector speed reduction.
- Turn 1 entry: Moderate harvest from front straight braking.

Battery management thresholds:
- Single DRS zone with medium harvest profile; moderate per-lap energy requirement.
- Nominal SOC entering main straight: 0.38+.
- SOC below 0.28: maintain full straight deployment; reduce earlier partial assists.
- SOC below 0.18: harvest priority at Turn 14; minimal deployment.
- SOC below 0.12: full harvest mode. Alert engineer.

Night race context: Qatar is run as an evening/night race. Ambient temperatures fall during the race (35°C to 28°C), which has a modest positive effect on battery thermal management — cooling conditions improve over the race duration. This is opposite to hot daytime circuits; the ERS system may be able to sustain higher deployment rates in the second stint than the first.
