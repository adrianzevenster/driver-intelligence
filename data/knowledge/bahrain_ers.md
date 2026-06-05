# Bahrain ERS Deployment
Bahrain has two clear DRS zones and consistent medium-speed corners where ERS deployment provides measurable lap time.

Key deployment zones:
- Main straight (Turns 15 to 1): Full deployment from Turn 15 exit through Turn 1 braking. Primary zone; adds 0.3–0.4 s per lap.
- Back straight (Turns 3–4): Secondary DRS zone. Deploy from Turn 3 exit; typically 60% deployment to manage battery for main straight.

Key harvest zones:
- Turn 10 hairpin braking: Strong regen on entry; highest per-corner SOC recovery on the circuit.
- Turn 1 braking: Good regen; combine with hairpin recovery to sustain main-straight deployment.

Battery management thresholds:
- SOC below 0.22 at Turn 14: reduce back-straight deployment to harvest-priority. Maintains main-straight push.
- SOC below 0.12: full harvest. Alert engineer. At Bahrain this is unusual; check deployment mapping.
- Night race note: track temperature drops 10–15°C from qualifying to race night, improving regen efficiency slightly. Recalibrate deployment schedule after lap 5 once temperatures stabilise.
