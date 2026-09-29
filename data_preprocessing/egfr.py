"""Estimated glomerular filtration rate (eGFR).

Implements the **2021 CKD-EPI creatinine equation (race-free)**, which is the
current recommended standard and avoids the race coefficient used by older
equations.

    eGFR = 142 * min(Scr/kappa, 1)**alpha
               * max(Scr/kappa, 1)**(-1.200)
               * 0.9938**Age
               * (1.012 if female else 1.0)

where Scr is serum creatinine in mg/dL, Age in years, and
    kappa = 0.7 (female) / 0.9 (male)
    alpha = -0.241 (female) / -0.302 (male)

Reference: Inker et al., NEJM 2021; KDIGO 2024 CKD guideline.
"""

import numpy as np


def ckd_epi_2021(scr_mg_dl, age_years, is_female):
    """Vectorised CKD-EPI 2021 eGFR (mL/min/1.73 m^2).

    Parameters may be scalars or numpy arrays of equal shape.
    """
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age_years, dtype=float)
    female = np.asarray(is_female, dtype=bool)

    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)

    ratio = scr / kappa
    egfr = (
        142.0
        * np.minimum(ratio, 1.0) ** alpha
        * np.maximum(ratio, 1.0) ** (-1.200)
        * 0.9938 ** age
        * np.where(female, 1.012, 1.0)
    )
    return egfr


def ckd_stage(egfr):
    """Map eGFR to a KDIGO G-stage label (string)."""
    egfr = np.asarray(egfr, dtype=float)
    stages = np.full(egfr.shape, "G?", dtype=object)
    stages[egfr >= 90] = "G1"
    stages[(egfr >= 60) & (egfr < 90)] = "G2"
    stages[(egfr >= 45) & (egfr < 60)] = "G3a"
    stages[(egfr >= 30) & (egfr < 45)] = "G3b"
    stages[(egfr >= 15) & (egfr < 30)] = "G4"
    stages[egfr < 15] = "G5"
    return stages


if __name__ == "__main__":
    # quick sanity check against published example values
    print("Female, Scr=0.9, age=60:", round(float(ckd_epi_2021(0.9, 60, True)), 1))
    print("Male,   Scr=1.2, age=60:", round(float(ckd_epi_2021(1.2, 60, False)), 1))
