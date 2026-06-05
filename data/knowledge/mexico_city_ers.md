# Mexico City ERS Deployment
Mexico City's high altitude fundamentally changes the ERS deployment profile. The MGU-K (motor) provides a proportionally larger fraction of total traction force because the ICE is working harder to compensate for oxygen-depleted combustion. ERS deployment is more valuable per unit SOC than at sea level.

Key deployment zones:
- Back straight (Turn 13 to Turn 16): Primary DRS zone; full deployment; reaches 360+ km/h at this altitude (equivalent to ~330 km/h at sea level). Adds approximately 0.55 s — highest single deployment value on the calendar.
- Start/finish straight (Turn 20 to Turn 1): Secondary zone; full deployment; adds approximately 0.35 s.

Key harvest zones:
- Turn 1 braking: Hard deceleration from 360+ km/h; excellent harvest but brakes run cooler; confirm optimal harvest temperature.
- Turn 6 (stadium hairpin) braking: Good harvest event.
- Turn 16 braking: Hard stop from maximum speed; best harvest event on circuit.

Battery management thresholds:
- Altitude reduces ICE contribution; ERS is effectively carrying a larger power share. More conservative SOC management to avoid peak depletion.
- Nominal SOC entering back straight: 0.52+.
- SOC below 0.38: reduce start/finish straight deployment; protect back straight.
- SOC below 0.25: back straight only; harvest priority at all braking zones.
- SOC below 0.12: full harvest mode. Alert engineer.

Altitude note: Battery temperature regulation differs at altitude. Cooler ambient air assists cooling but reduces atmospheric pressure, which changes heat transfer coefficients. Monitor battery temperature against altitude-specific baseline, not standard sea-level thresholds.
