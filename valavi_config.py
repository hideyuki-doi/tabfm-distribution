# Shared settings for the Valavi et al. (2022)-compatible benchmark (v40).
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "Records")
REGIONS = ["AWT", "CAN", "NSW", "NZ", "SA", "SWI"]

# Uncorrelated covariates per region (|r| <= 0.8), from Valavi's code
# (github.com/rvalavi/SDM-Spatial-Validation, R/nceas_model_vars.R). NSW includes vegsys,
# as in the e01486 Appendix S2 example (it is commented out only in the later paper).
VARS = {
    "AWT": ["bc04", "bc05", "bc06", "bc12", "bc15", "slope", "topo", "tri"],
    "CAN": ["alt", "asp2", "ontprec", "ontslp", "onttemp", "ontveg", "watdist"],
    "NSW": ["cti", "disturb", "mi", "rainann", "raindq", "rugged", "soildepth", "soilfert",
            "solrad", "tempann", "topo", "vegsys"],
    "NZ": ["age", "deficit", "hillshade", "mas", "mat", "r2pet", "slope", "sseas", "toxicats", "tseas", "vpd"],
    "SA": ["sabio12", "sabio15", "sabio17", "sabio18", "sabio2", "sabio4", "sabio5", "sabio6"],
    "SWI": ["bcc", "calc", "ccc", "ddeg", "nutri", "pday", "precyy", "sfroyy", "slope", "sradyy", "swb", "topo"],
}
# Categorical covariates (factors in Valavi's R code; Elith et al. 2020 metadata)
CATEGORICAL = {"CAN": ["ontveg"], "NSW": ["vegsys"], "NZ": ["age", "toxicats"], "SWI": ["calc"]}
