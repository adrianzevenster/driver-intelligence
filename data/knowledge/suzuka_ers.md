# Suzuka ERS Deployment
Suzuka has moderate ERS value. The S-curves and 130R are not deployment zones — they require driver focus and smooth inputs, and ERS intervention changes power delivery characteristics. The key zones are the overtaking sections.

Key deployment zones:
- Back straight (Spoon exit to Casio chicane): Primary deployment. Long straight with DRS; deploy from Spoon exit through chicane braking. Adds 0.3–0.4 s per lap.
- Start-finish straight (Casio exit to first curve): Full deployment lap start and restart situations; deploy for maximum first-corner positioning gain.
- Hairpin exit to Degner 2: Secondary deployment. Assists traction on the hairpin exit where wheel-spin on high-degradation rear tires is a risk.

Key harvest zones:
- Hairpin braking: Good regen on hairpin entry from the back straight. Typically 15–18% SOC per lap.
- Casio chicane: Regen from chicane braking; moderate yield.

Battery management thresholds:
- SOC below 0.25 at Spoon exit: reduce back-straight deployment to 60% to conserve for next lap's S-curves approach. The S-curves themselves do not use ERS deployment but a low-SOC entry to sector 1 means no push available at the start-finish for the following lap.
- SOC below 0.15: full harvest. Alert engineer. Avoid deployment at 130R under any circumstances if battery is critically low — power fluctuation mid-corner is dangerous.

Intervention notes: Suzuka rewards consistent ERS management across the full stint. Aggressive opening lap deployment at the expense of race-long battery availability is a common error on medium tire compounds.
