# Singapore ERS Deployment
Singapore is a medium-ERS-value circuit. The stop-start nature of the street circuit means constant deployment and harvest cycling. The heat creates battery thermal management demands not seen at cooler circuits.

Key deployment zones:
- Marina Bay straight (Turns 5–7): Primary deployment on the circuit's only meaningful straight. Full push from Turn 5 exit through Turn 7 braking.
- Pit straight / main straight: Deployment for final push before braking zone in sector 1. DRS-assisted.

Key harvest zones:
- Turn 1, Turn 10, Turn 14, Turn 23: Multiple heavy braking zones yield consistent regen throughout the lap. Singapore has one of the highest regen-per-lap rates on the calendar due to constant braking events.

Battery management thresholds:
- SOC should remain above 0.35 throughout the race due to the high harvest rate. If SOC falls below 0.30, a harvest mapping failure is likely — alert engineer.
- Battery thermal management: Singapore ambient temperature (32°C, 85% humidity) means battery operates near thermal ceiling. If battery temperature warning activates, reduce deployment to 50% immediately and sustain for at least 3 laps. At Singapore this costs approximately 0.5 s/lap versus loss of power unit.
- Safety car mode: under safety car, maximum harvest is standard. Do not deploy under safety car unless restart overlap requires it.
