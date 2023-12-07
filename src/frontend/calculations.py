def calculate_ckd_epi_gfr(age, serum_creatinine, is_female=False, is_black=False):
    if is_female:
        kappa = 61.9
        alpha = -0.329
    else:
        kappa = 79.6
        alpha = -0.411
    
    if is_black:
        race_factor = 1.159
    else:
        race_factor = 1.0
    
    gfr = 141 * min(serum_creatinine / kappa, 1) ** alpha * max(serum_creatinine / kappa, 1) ** -1.209 * 0.993 ** age * 1.018 ** is_female * race_factor
    
    return gfr

def calculate_bmi(weight, height):
    return weight / ( (height/100) **2)