# Zandvoort ERS Deployment
Zandvoort has a single DRS zone and a moderate ERS profile. The unique banked final corner (Turn 14) means the car arrives at high speed onto the main straight with minimal braking; this reduces the harvest opportunity before the DRS zone compared to circuits with heavy final-corner braking.

Key deployment zones:
- Main straight (Turn 14 exit to Turn 1): Single DRS zone; full deployment from banking exit; reaches ~300 km/h. Adds approximately 0.3 s.

Key harvest zones:
- Turn 1 (Tarzan) braking: Hardest braking event on the circuit; deceleration from 290+ km/h; best harvest.
- Turns 5–6 braking: Medium deceleration; moderate harvest.
- Hugenholtz complex (Turns 8–9): Some light deceleration for harvest.

Battery management thresholds:
- Single DRS zone with modest top speed means lower per-lap consumption than multi-zone circuits.
- Nominal SOC entering main straight: 0.38+.
- SOC below 0.28: no deployment change needed; maintain full straight deployment — it is the only opportunity.
- SOC below 0.15: harvest priority at Turn 1; minimal deployment.
- SOC below 0.12: full harvest mode. Alert engineer.

Wind note: Strong coastal crosswinds create unpredictable aerodynamic variations. If the car is unstable through Turn 14 banking due to wind, reducing deployment on exit from that corner reduces throttle aggressiveness and may prevent snap oversteer. Coordinate with driver on wind conditions per lap.
