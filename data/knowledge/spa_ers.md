# Spa-Francorchamps ERS Deployment
Spa has two high-value ERS deployment zones and one major harvest zone. The circuit's elevation changes affect brake-regen yield significantly.

Key deployment zones:
- Kemmel Straight (Les Combes approach): Primary deployment. Full push from Eau Rouge exit to Les Combes braking. Adds 0.4–0.5 s per lap. DRS zone active here.
- Blanchimont to Bus Stop: Secondary deployment. Driver can push 70% from Blanchimont exit through the chicane approach for final sector time.

Key harvest zones:
- La Source braking: Strong regen from Kemmel arrival into La Source hairpin. Typically yields 15–20% SOC per lap.
- Raidillon approach: Medium regen available; do not sacrifice pace here for SOC.

Battery management thresholds:
- SOC below 0.25 at Eau Rouge exit: reduce Kemmel deployment to 70% and harvest through Les Combes. Net lap time cost ~0.2 s but maintains deployment availability for final push.
- SOC below 0.15: full harvest mode. Alert engineer.
- In wet conditions, regen yield at La Source drops 30% (lower brake energy on entry); recalibrate deployment schedule accordingly.

Intervention notes: Spa weather changes can force a sudden mode switch. If intermediate tires are fitted, deployment profiles should be reduced by 20% to avoid oversteer on cold rubber mid-lap.
