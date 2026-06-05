# Imola ERS Deployment
Imola has one DRS zone (start/finish straight) and a moderate ERS harvest profile. The anti-clockwise direction does not change ERS deployment logic but does change the sectors in which harvest events occur.

Key deployment zones:
- Main straight (Rivazza exit to Tamburello): Single DRS zone; full deployment; adds approximately 0.3 s.

Key harvest zones:
- Tamburello braking: Medium-high harvest from fast approach.
- Tosa hairpin braking: Best harvest event on circuit; hard deceleration from 260 km/h.
- Variante Alta chicane: Good harvest from mid-sector braking.

Battery management thresholds:
- Single DRS zone means lower per-lap energy demand than multi-zone circuits.
- Nominal SOC entering main straight: 0.35+.
- SOC below 0.25: maintain full main straight deployment; skip Acque Minerali sector partial boost.
- SOC below 0.15: harvest priority at Tosa and Variante Alta; skip all non-DRS deployment.
- SOC below 0.12: full harvest. Alert engineer.

Narrow circuit note: With limited overtaking, ERS deployment for defensive positioning (matching a rival's DRS activation on the straight) is more tactically important here than at wide-overtaking circuits. Ensure SOC is ≥0.40 when defending from a faster car behind.
