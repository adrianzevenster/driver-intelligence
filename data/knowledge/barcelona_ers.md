# Barcelona ERS Deployment
Barcelona has two primary deployment zones and is a medium-ERS-value circuit. The combination of a long main straight and a back straight in sector 2 rewards consistent deployment scheduling.

Key deployment zones:
- Main straight (Turn 16 to Turn 1): Full deployment from final hairpin exit. DRS zone. Adds 0.3–0.4 s per lap. Primary zone.
- Back straight (Turns 7–9, Campsa approach): Secondary deployment from Turn 7 exit to Campsa entry. DRS zone. Adds 0.2 s.

Key harvest zones:
- Turn 1 braking: Good regen from high-speed entry; primary recovery source.
- Turn 10 chicane braking: Medium regen; supplements main harvest.

Battery management thresholds:
- SOC below 0.22 at Turn 15 (final chicane entry): reduce back-straight deployment to harvest mode. Maintain full deployment on the main straight only.
- SOC below 0.12: full harvest. Alert engineer.
- High track temperature note (May, track 45–50°C): Battery thermal management is a consideration at Barcelona. If battery temperature exceeds operating range, a deployment reduction of 15% is standard protocol; alert engineer to check thermal status.
