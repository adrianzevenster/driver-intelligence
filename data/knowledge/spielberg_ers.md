# Red Bull Ring ERS Deployment
Despite being the shortest circuit, the Red Bull Ring has two DRS zones and delivers one of the highest ERS return-on-investment per lap of any circuit due to two strong harvest events at heavy braking zones.

Key deployment zones:
- Main straight (Turn 9 exit to Turn 1): Primary DRS zone; full deployment from hairpin exit; reaches 310 km/h. Adds approximately 0.4 s.
- Back straight (Turn 3 exit to Turn 4): Secondary DRS zone; adds approximately 0.2 s.

Key harvest zones:
- Turn 3 (Remus) braking: Downhill deceleration from 300 km/h; excellent harvest — one of the best single events on the calendar.
- Turn 9 (Rindt Corner) braking: Hard stop into the final hairpin; strong harvest.
- Turn 6 medium braking: Moderate harvest in the elevation change section.

Battery management thresholds:
- Short lap means more deployment events per unit time than longer circuits; SOC cycles faster.
- Nominal SOC entering main straight: 0.42+.
- SOC below 0.30: disable back straight DRS deployment; protect main straight.
- SOC below 0.20: main straight only; harvest priority at Turns 3 and 9.
- SOC below 0.12: full harvest mode. Alert engineer.

Altitude context: At 660m elevation, air density is ~8% lower. MGU-H (heat-side energy recovery from turbo) is modestly reduced, which can affect peak SOC recovery rates in longer stints. Monitor whether battery charge rate slows relative to sea-level reference data.
