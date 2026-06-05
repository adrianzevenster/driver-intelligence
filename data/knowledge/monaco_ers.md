# Monaco ERS Deployment
Monaco is the lowest ERS-value circuit on the calendar due to very low straight-line speeds and abundant brake-regen from constant braking. Battery is rarely the limiting factor.

Key deployment zones:
- Tunnel exit to Nouvelle Chicane: Only meaningful straight. Deploy from mid-tunnel to chicane braking; adds 0.2 s at most.
- Beau Rivage climb: Light deployment assists acceleration from Sainte Dévote. Modest gain.
- Anthony Noghes to Sainte Dévote: Deployment at the pit-straight for any following attempt; almost entirely symbolic at Monaco — no overtaking is feasible regardless.

Key harvest zones:
- Loews hairpin: Significant regen from the slowest corner in F1. Typically yields 20–25% SOC per lap.
- Sainte Dévote, Mirabeau, Casino, Rascasse: Constant braking means harvest is abundant throughout.

Battery management thresholds:
- SOC rarely drops below 0.40 at Monaco due to constant regen. If SOC falls below 0.30, check for harvest failure.
- ERS deployment mode: recommend conservative "balanced" mode throughout. Full push mode is wasteful and yields negligible lap time improvement.

Intervention notes: At Monaco, ERS monitoring should be deprioritised. Tire condition, flat-spots, and undercut windows are the dominant decision variables.
