import pathlib
import pandas as pd

from sklearn.preprocessing import StandardScaler


# DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"

DATA_DIR = pathlib.Path('../uci-tabular/uciml/data/').parent.resolve() / "data"


# Dict of dataset name to label target, used to convert the 'label' column to 0/1 vector
targets = {
    "adult": ">50K",
    "annealing": "3",
    "audiology-std": 1,
    "bank": "yes",
    "bankruptcy": "B",
    "car": "unacc",
    "chess-krvk": "draw",
    "chess-krvkp": "won",
    "congress-voting": "democrat",
    # Contraceptive method choice. Target: Contraceptive method used (no-use, long-term, short-term)
    "contrac": 1,
    # Credit card applications. Target: Approved.
    "credit-approval": "+",
    # Cardiotocography. Target: Fetal state (normal, suspect, pathologic)
    "ctg": 1,
    "cylinder-bands": "noband",
    # Differential diagnosis of erythemato-squamous diseases. Target: disease (psoriasis, seboreic
    # dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris)
    "dermatology": 1,
    "german-credit": 1,
    "heart-cleveland": range(1, 5 + 1),
    "ilpd": 1,
    # Mammographic mass. Target: severity (benign, malignant)
    "mammo": 1,
    "mushroom": "p",
    "wine": 2,
    "wine-qual": range(6, 9 + 1),
}

# Dict of dataset name to column names, used to build the pandas DataFrames
columns = {
    "adult": [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_no",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "label",
    ],
    "annealing": [
        "family",
        "product",
        "steel",
        "carbon",
        "hardness",
        "temper_rolling",
        "condition",
        "formability",
        "strength",
        "non",
        "surface_finish",
        "surface_quality",
        "enamelability",
        "bc",
        "bf",
        "bt",
        "bw",
        "bl",
        "m",
        "chrom",
        "phos",
        "cbond",
        "marvi",
        "exptl",
        "ferro",
        "corr",
        "blue",
        "lustre",
        "jurofm",
        "s",
        "p",
        "shape",
        "thick",
        "width",
        "len",
        "oil",
        "bore",
        "packing",
        "label",
    ],
    "audiology-std": [
        "label",
        "age_gt_60",
        "air",
        "airBoneGap",
        "ar_c",
        "ar_u",
        "boneAbnormal",
        "history_dizziness",
        "history_fluctuating",
        "history_nausea",
        "history_noise",
        "history_recruitment",
        "history_ringing",
        "history_roaring",
        "history_vomiting",
        "late_wave_poor",
        "m_m_gt_2k",
        "m_m_sn",
        "m_m_sn_gt_1k",
        "m_m_sn_gt_2k",
        "m_sn_gt_1k",
        "m_sn_gt_2k",
        "m_sn_gt_3k",
        "m_sn_gt_4k",
        "m_sn_lt_1k",
        "middle_wave_poor",
        "mod_sn_gt_2k",
        "mod_sn_gt_3k",
        "mod_sn_gt_4k",
        "mod_sn_gt_500",
        "notch_4k",
        "notch_at_4k",
        "o_ar_c",
        "o_ar_u",
        "s_sn_gt_1k",
        "s_sn_gt_2k",
        "s_sn_gt_4k",
        "speech",
        "static_normal",
        "tymp",
        "wave_V_delayed",
        "waveform_ItoV_prolonged",
    ],
    "bank": [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
        "label",
    ],
    "bankruptcy": [
        "industrial_risk",
        "management_risk",
        "financial_flex",
        "credibility",
        "competitiveness",
        "operating_risk",
        "label",
    ],
    "car": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"],
    "chess-krvk": [
        "White_King_file",
        "White_King_rank",
        "White_Rook_file",
        "White_Rook_rank",
        "Black_King_file",
        "Black_King_rank",
        "label",
    ],
    "chess-krvkp": [
        "bkblk",
        "bknwy",
        "bkon8",
        "bkona",
        "bkspr",
        "bkxbq",
        "bkxcr",
        "bkxwp",
        "blxwp",
        "bxqsq",
        "cntxt",
        "dsopp",
        "dwipd",
        "hdchk",
        "katri",
        "mulch",
        "qxmsq",
        "r2ar8",
        "reskd",
        "reskr",
        "rimmx",
        "rkxwp",
        "rxmsq",
        "simpl",
        "skach",
        "skewr",
        "skrxp",
        "spcop",
        "stlmt",
        "thrsk",
        "wkcti",
        "wkna8",
        "wknck",
        "wkovl",
        "wkpos",
        "wtoeg",
        "label",
    ],
    "congress-voting": [
        "label",
        "handicapped_infants",
        "water_project_cost_sharing",
        "adoption_of_the_budget_resolution",
        "physician_fee_freeze",
        "el_salvador_aid",
        "religious_groups_in_schools",
        "anti_satellite_test_ban",
        "aid_to_nicaraguan_contras",
        "mx_missile",
        "immigration",
        "synfuels_corporation_cutback",
        "education_spending",
        "superfund_right_to_sue",
        "crime",
        "duty_free_exports",
        "export_administration_act_south_africa",
    ],
    "contrac": [
        "age",
        "wifes_education",
        "husbands_education",
        "nchildren",
        "religion",
        "working",
        "husbands_occupation",
        "living_std",
        "media_exposure",
        "label",
    ],
    "credit-approval": ["A" + str(i) for i in range(1, 15 + 1)] + ["label"],
    "ctg": [
        "Date",
        "b",
        "e",
        "LBE",
        "LB",
        "AC",
        "FM",
        "UC",
        "ASTV",
        "mSTV",
        "ALTV",
        "mLTV",
        "DL",
        "DS",
        "DP",
        "DR",
        "Width",
        "Min",
        "Max",
        "Nmax",
        "Nzeros",
        "Mode",
        "Mean",
        "Median",
        "Variance",
        "Tendency",
        "A",
        "B",
        "C",
        "D",
        "SH",
        "AD",
        "DE",
        "LD",
        "FS",
        "SUSP",
        "CLASS",
        "label",
    ],
    "cylinder-bands": [
        "timestamp",
        "cylinder_number",
        "customer",
        "job_number",
        "grain_screened",
        "ink_color",
        "proof_on_ctd_ink",
        "blade_mfg",
        "cylinder_division",
        "paper_type",
        "ink_type",
        "direct_steam",
        "solvent_type",
        "type_on_cylinder",
        "press_type",
        "press",
        "unit_number",
        "cylinder_size",
        "paper_mill_location",
        "plating_tank",
        "proof_cut",
        "viscosity",
        "caliper",
        "ink_temperature",
        "humifity",
        "roughness",
        "blade_pressure",
        "varnish_pct",
        "press_speed",
        "ink_pct",
        "solvent_pct",
        "ESA_Voltage",
        "ESA_Amperage",
        "wax",
        "hardener",
        "roller_durometer",
        "current_density",
        "anode_space_ratio",
        "chrome_content",
        "label",
    ],
    "dermatology": [
        "erythema",
        "scaling",
        "definite_borders",
        "itching",
        "koebner_phenomenon",
        "polygonal_papules",
        "follicular_papules",
        "oral_mucosal_involvement",
        "knee_and_elbow_involvement",
        "scalp_involvement",
        "family_history",
        "melanin_incontinence",
        "eosinophils_in_the_infiltrate",
        "PNL_infiltrate",
        "fibrosis_of_the_papillary_dermis",
        "exocytosis",
        "acanthosis",
        "hyperkeratosis",
        "parakeratosis",
        "clubbing_of_the_rete_ridges",
        "elongation_of_the_rete_ridges",
        "thinning_of_the_suprapapillary_epidermis",
        "spongiform_pustule",
        "munro_microabcess",
        "focal_hypergranulosis",
        "disappearance_of_the_granular_layer",
        "vacuolisation_and_damage_of_basal_layer",
        "spongiosis",
        "saw_tooth_appearance_of_retes",
        "follicular_horn_plug",
        "perifollicular_parakeratosis",
        "inflammatory_monoluclear_inflitrate",
        "band_like_infiltrate",
        "age",
        "label",
    ],
    "german-credit": [
        "Status_of_checking_account",
        "Duration_in_months",
        "Credit_history",
        "Purpose",
        "Credit_amount",
        "Savings_account",
        "Employment_since",
        "Installment_rate",
        "status_and_sex",
        "Other_debtors",
        "residence_since",
        "Property",
        "Age",
        "Other_installment",
        "Housing",
        "existing_credits",
        "Job",
        "Number_of_dependents",
        "Have_Telephone",
        "foreign_worker",
        "label",
    ],
    "heart-cleveland": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "label",
    ],
    "ilpd": [
        "age",
        "gender",
        "tb",
        "db",
        "alkphos",
        "sgpt",
        "sgot",
        "tp",
        "alb",
        "ag",
        "label",
    ],
    "mammo": ["BIRADS", "age", "shape", "margin", "density", "label"],
    "mushroom": [
        "label",
        "cap_shape",
        "cap_surface",
        "cap_color",
        "bruises",
        "odor",
        "gill_attachment",
        "gill_spacing",
        "gill_size",
        "gill_color",
        "stalk_shape",
        "stalk_root",
        "stalk_surface_above",
        "stalk_surface_below",
        "stalk_color_above",
        "stalk_color_below",
        "veil_type",
        "veil_color",
        "ring_number",
        "ring_type",
        "spore_print_color",
        "population",
        "habitat",
    ],
    "wine": [
        "label",
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315",
        "Proline",
    ],
    "wine-qual": [
        "fixed.acidity",
        "volatile.acidity",
        "citric.acid",
        "residual.sugar",
        "chlorides",
        "free.sulfur.dioxide",
        "total.sulfur.dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "label",
        "color",
    ],
}

# Make sure that the two dictionaries have the same keys
assert (
    columns.keys() == targets.keys()
), "something is wrong with columns and targets dictionaries"


# Taken from https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
german_mappings = {
    "Status_of_checking_account": {
        "A11": "... < 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": "... >= 200 DM / salary assignments for at least 1 year",
        "A14": "no checking account",
    },
    "Credit_history": {
        "A30": "no credits taken/ all credits paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account/ other credits existing (not at this bank)",
    },
    "Purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "(vacation - does not exist?)",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "Savings_account": {
        "A61": "... < 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ".. >= 1000 DM",
        "A65": "unknown/ no savings account",
    },
    "Employment_since": {
        "A71": "unemployed",
        "A72": "... < 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ".. >= 7 years",
    },
    "status_and_sex": {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single",
        "A94": "male : married/widowed",
        "A95": "female : single",
    },
    "Other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "Property": {
        "A121": "real estate",
        "A122": "if not A121 : building society savings agreement/ life insurance",
        "A123": "if not A121/A122 : car or other, not in attribute 6",
        "A124": "unknown / no property",
    },
    "Other_installment": {"A141": "bank", "A142": "stores", "A143": "none"},
    "Housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "Job": {
        "A171": "unemployed/ unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee / official",
        "A174": "management/ self-employed/ highly qualified employee/ officer",
    },
    "Have_Telephone": {
        "A191": "none",
        "A192": "yes, registered under the customers name",
    },
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def load_texas(one_hot_encoded=True, dropna=False):
    """
    Loads Texas Hospital Discharge dataset.
    Code is based on a script by Theresa Stadler.
    Returns:
        X: DataFrame with the features
        y: DataFrame with the label, values are only 0 and 1
    """
    TEXAS_PATH = DATA_DIR / "texas/PUDF_base1_{}q2013_tab.txt"

    cat_cols = [
        "TYPE_OF_ADMISSION",
        "PAT_STATE",
        "PAT_AGE",
        "PAT_STATUS",
        "SEX_CODE",
        "RACE",
        "ETHNICITY",
        "ILLNESS_SEVERITY",
        "RISK_MORTALITY",
    ]

    num_cols = ["LENGTH_OF_STAY", "TOTAL_CHARGES", "TOTAL_NON_COV_CHARGES"]

    dtypes = {
        **{col: "category" for col in cat_cols},
        **{col: "float" for col in num_cols},
    }

    # Read each of the 4 files of the dataset and put them in a list
    q = []
    for i in range(4):
        df = pd.read_csv(
            TEXAS_PATH.format(i + 1),
            delimiter="\t",
            dtype=dtypes,
            usecols=cat_cols + num_cols,
            na_values=["`"],
        )
        q.append(df)
        print(f"Loaded {i+1} / 4", end="\r")
    print()

    # Merge the DataFrames into a single one
    combined_df = pd.concat(q, ignore_index=True)

    # Only keep the first 3 digits of zip (padded with zeroes if the original is shorter than 5 digits)
    # combined_df['PAT_ZIP'] = combined_df['PAT_ZIP'].map(lambda z: z.zfill(5)[:2], na_action='ignore')

    # We define the label to be whether there were non-covered charges or not
    X = combined_df.drop(columns="TOTAL_NON_COV_CHARGES")
    y = (combined_df["TOTAL_NON_COV_CHARGES"] > 0).astype(int)

    return X, y


def one_hot_encode(df, sep="~"):
    """
    One-hot encodes the given dataset, by transforming each categorical feature column into
     a set of binary features, and grouping them together in a pandas MultiIndex.
    If a column contains NaNs, then the dummy DataFrame will have a NaN in the second level as well.

    Args:
        df : DataFrame to one-hot encode
        sep: String to be used for the construction of the MultiIndex, can't appear in the columns

    Returns:
        new_df: DataFrame with a MultiIndex column index, containing only binary features
    """
    # Initialize the DataFrame
    new_dfs = {}
    for col in df.columns:
        # Dummy column, and if this column contains nan, create a dummy column for nans as well
        if df[col].dtype in ["categorical", "object"]:
            temp_df = pd.get_dummies(
                df[[col]], columns=[col], prefix_sep=sep, dummy_na=df[col].isna().any()
            )
            for cat in temp_df.columns:
                new_dfs[cat] = temp_df[cat]
        else:
            new_dfs[col] = df[col]

    result = pd.DataFrame.from_dict(new_dfs)
    # result.columns = pd.MultiIndex.from_tuples([c.split(sep) for c in new_dfs.keys()])
    return result


def load_dataset(name, one_hot_encoded=True, dropna=False, norm=True):
    """
    Loads the dataset with the given name, and returns two DataFrames with the features and label.

    Args:
        name: string name of the dataset to load, without the '.csv' extension.
        one_hot_encoded (bool): whether to one-hot encode categorical variables.
        dropna (bool): Whether to drop NAN rows
        norm: Whether to normalize the numeric features.

    Returns:
        X: DataFrame with the features
        y: DataFrame with the label, values are only 0 and 1
    """
    if name == "texas":
        return load_texas(one_hot_encoded=one_hot_encoded, dropna=dropna)
    elif name not in targets.keys():
        raise ValueError(f"Unknown dataset: {name}")

    # Load the data from the csv file
    df = pd.read_csv(
        DATA_DIR / f"{name}.csv",
        sep=",",
        skipinitialspace=True,
        header=None,
        names=columns[name],
        na_values=["?"],
    )

    if dropna:
        df = df.dropna()

    # We want the target label/s to be 1, and the other/s 0
    if isinstance(targets[name], range):
        df["label"] = (df["label"].isin(targets[name])).astype(int)
    else:
        df["label"] = (df["label"] == targets[name]).astype(int)

    # Scale the numeric features. Admits some test data leakage.
    if norm:
        numeric_cols = [
            col for col in df.select_dtypes(["int", "float"]).columns if col != "label"
        ]
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
        df[numeric_cols] = scaled_values

    # If the dataset is german, replace the encoded categories by their meaning
    if name == "german-credit":
        df = df.replace(german_mappings)

    # Create the dataframe of features
    X = df.drop(columns=["label"])
    y = df["label"]

    if one_hot_encoded:
        X = one_hot_encode(X)

    return X, y
