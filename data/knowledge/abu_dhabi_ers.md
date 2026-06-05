# Abu Dhabi ERS Deployment
Abu Dhabi has three DRS zones — the highest count of any circuit. ERS management across three deployment zones requires careful SOC scheduling to avoid depletion mid-race.

Key deployment zones:
- Main straight (Turn 21 to Turn 1): Primary zone. Full deployment from final corner exit. Longest straight on the circuit; adds 0.4 s.
- Sector 2 straight (Turn 9 to Turn 11): Secondary zone. DRS from Turn 9 exit to Turn 11 hairpin braking. Adds 0.25 s.
- Inner straight (Turn 12 to Turn 14): Tertiary zone. Shorter; adds 0.15 s. Consider reducing deployment here to preserve battery for main straight when SOC is marginal.

Key harvest zones:
- Turn 1 braking: Primary harvest; high-speed entry good regen.
- Turn 11 hairpin: Secondary harvest.
- Turns 13–14 chicane: Light regen.

Battery management thresholds:
- With three DRS zones, SOC budget is tighter than two-zone circuits. Nominal SOC entering main straight: 0.45+ for full deployment across all three zones.
- SOC below 0.30: disable tertiary zone (Turn 12–14 straight) deployment. Maintain main and secondary.
- SOC below 0.20: disable secondary zone deployment. Main straight only.
- SOC below 0.12: full harvest. Alert engineer.

Season finale context: Abu Dhabi closes the season. Power unit conservation (protecting against grid penalties for following-year components) may reduce deployment ceiling. Confirm with engineer whether power unit mode restrictions apply.
