# Miami ERS Deployment
Miami has three DRS zones, making ERS management moderately complex. The circuit speed profile is mixed, limiting harvest efficiency in slow sectors.

Key deployment zones:
- Main straight (Turn 19 to Turn 1): Primary zone; full deployment from pit exit; reaches ~300 km/h. Adds 0.4 s.
- Back straight (Turn 12 to Turn 14): Secondary zone; full deployment; adds 0.3 s.
- Pit straight approach (Turn 16 exit): Tertiary zone; shorter, adds 0.15 s.

Key harvest zones:
- Turn 1 braking: Hard deceleration from 280 km/h; high harvest.
- Turn 4 hairpin braking: Medium harvest.
- Turn 11 braking: Heavy braking from back straight; good harvest event.

Battery management thresholds:
- Nominal SOC entering main straight: 0.42+.
- SOC below 0.32: reduce tertiary zone deployment; protect main and secondary.
- SOC below 0.22: disable secondary deployment; main straight only.
- SOC below 0.12: full harvest. Alert engineer.

Heat context: Miami in May has high ambient temperature (28–32°C) and humidity, which stresses the battery thermal management system. Monitor battery temperature alongside SOC — thermal derating may force early deployment reduction even at adequate SOC.
