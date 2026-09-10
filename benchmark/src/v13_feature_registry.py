from __future__ import annotations

from typing import Any


DYNAMIC_FEATURES = (
    "heart_rate", "sbp", "dbp", "map", "spo2", "resp_rate", "temp_c", "temp_f",
    "gcs_total", "gcs_verbal", "gcs_motor", "gcs_eye", "glucose", "fio2",
    "urine_output", "lactate", "bicarbonate", "ph", "pao2", "paco2", "creatinine",
    "bun", "sodium", "potassium", "chloride", "calcium", "magnesium", "phosphate",
    "hemoglobin", "hematocrit", "wbc", "platelet", "bilirubin_total", "alt", "ast",
    "inr", "pt", "ptt", "crp", "procalcitonin",
)

ELIXHAUSER_FEATURES = (
    "chf", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation", "pvd",
    "hypertension", "paralysis", "other_neurological", "copd",
    "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "renal_failure",
    "liver_disease", "peptic_ulcer", "aids", "lymphoma", "metastatic_cancer",
    "solid_tumor", "rheumatoid_arthritis", "coagulopathy", "obesity", "weight_loss",
    "fluid_electrolyte", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse",
    "drug_abuse", "psychoses", "depression",
)

STATIC_FEATURES = (
    "age", "gender_male", "emergency_admit", "elective_admit", "insurance_medicare",
    "insurance_medicaid", "insurance_private", "micu", "sicu", "cicu", "nicu", "csru",
)


DYNAMIC_METADATA: dict[str, tuple[str, str, str]] = {
    "heart_rate": ("Heart rate", "beats/min", "vital sign"),
    "sbp": ("Systolic blood pressure", "mmHg", "vital sign"),
    "dbp": ("Diastolic blood pressure", "mmHg", "vital sign"),
    "map": ("Mean arterial pressure", "mmHg", "vital sign"),
    "spo2": ("Peripheral oxygen saturation", "%", "respiratory"),
    "resp_rate": ("Respiratory rate", "breaths/min", "respiratory"),
    "temp_c": ("Temperature", "degrees C", "vital sign"),
    "temp_f": ("Temperature", "degrees F", "vital sign"),
    "gcs_total": ("Glasgow Coma Scale total", "score", "neurologic"),
    "gcs_verbal": ("Glasgow Coma Scale verbal", "score", "neurologic"),
    "gcs_motor": ("Glasgow Coma Scale motor", "score", "neurologic"),
    "gcs_eye": ("Glasgow Coma Scale eye", "score", "neurologic"),
    "glucose": ("Glucose", "mg/dL", "chemistry"),
    "fio2": ("Fraction of inspired oxygen", "%", "respiratory"),
    "urine_output": ("Urine output", "mL per charted interval", "fluid balance"),
    "lactate": ("Lactate", "mmol/L", "blood gas"),
    "bicarbonate": ("Bicarbonate", "mEq/L", "chemistry"),
    "ph": ("pH", "pH units", "blood gas"),
    "pao2": ("Arterial oxygen partial pressure", "mmHg", "blood gas"),
    "paco2": ("Arterial carbon dioxide partial pressure", "mmHg", "blood gas"),
    "creatinine": ("Creatinine", "mg/dL", "chemistry"),
    "bun": ("Blood urea nitrogen", "mg/dL", "chemistry"),
    "sodium": ("Sodium", "mEq/L", "chemistry"),
    "potassium": ("Potassium", "mEq/L", "chemistry"),
    "chloride": ("Chloride", "mEq/L", "chemistry"),
    "calcium": ("Total calcium", "mg/dL", "chemistry"),
    "magnesium": ("Magnesium", "mg/dL", "chemistry"),
    "phosphate": ("Phosphate", "mg/dL", "chemistry"),
    "hemoglobin": ("Hemoglobin", "g/dL", "hematology"),
    "hematocrit": ("Hematocrit", "%", "hematology"),
    "wbc": ("White blood cell count", "10^3/uL", "hematology"),
    "platelet": ("Platelet count", "10^3/uL", "hematology"),
    "bilirubin_total": ("Total bilirubin", "mg/dL", "chemistry"),
    "alt": ("Alanine aminotransferase", "U/L", "chemistry"),
    "ast": ("Aspartate aminotransferase", "U/L", "chemistry"),
    "inr": ("International normalized ratio", "ratio", "coagulation"),
    "pt": ("Prothrombin time", "s", "coagulation"),
    "ptt": ("Partial thromboplastin time", "s", "coagulation"),
    "crp": ("C-reactive protein", "mg/L", "inflammation"),
    "procalcitonin": ("Procalcitonin", "ng/mL", "inflammation"),
}


MIMIC_CHART_ITEMS: dict[str, tuple[int, ...]] = {
    "heart_rate": (211, 220045),
    "sbp": (51, 442, 455, 6701, 220050, 220179, 225309),
    "dbp": (8368, 8440, 8441, 8555, 220051, 220180, 225310),
    "map": (52, 443, 456, 6702, 220052, 220181, 225312),
    "spo2": (646, 220227, 220277),
    "resp_rate": (615, 618, 619, 220210, 224688, 224689, 224690),
    "temp_c": (676, 677, 223762),
    "temp_f": (678, 679, 223761),
    "gcs_total": (198,),
    "gcs_verbal": (723, 223900),
    "gcs_motor": (454, 223901),
    "gcs_eye": (184, 220739),
    "glucose": (807, 811, 1310, 1455, 1529, 3744, 3745, 220621, 225664, 226537),
    "fio2": (190, 3420, 3422, 223835),
}

MIMIC_LAB_ITEMS: dict[str, tuple[int, ...]] = {
    "lactate": (50813,),
    "bicarbonate": (50803, 50882),
    "ph": (50820,),
    "pao2": (50821,),
    "paco2": (50818,),
    "creatinine": (50912,),
    "bun": (51006,),
    "sodium": (50824, 50983),
    "potassium": (50822, 50971),
    "chloride": (50806, 50902),
    "calcium": (50893,),
    "magnesium": (50960,),
    "phosphate": (50970,),
    "hemoglobin": (50811, 51222),
    "hematocrit": (50810, 51221),
    "wbc": (51300, 51301),
    "platelet": (51265,),
    "bilirubin_total": (50885,),
    "alt": (50861,),
    "ast": (50878,),
    "inr": (51237,),
    "pt": (51274,),
    "ptt": (51275,),
    "crp": (50889,),
    "procalcitonin": (50976,),
}

MIMIC_URINE_OUTPUT_ITEMS = (
    40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056,
    226559, 226560, 226561, 226563, 226564, 226565, 226566, 226584,
    226627, 226631, 227489,
)

EICU_PERIODIC_COLUMNS = {
    "temperature": "temp_c",
    "sao2": "spo2",
    "heartrate": "heart_rate",
    "respiration": "resp_rate",
    "systemicsystolic": "sbp",
    "systemicdiastolic": "dbp",
    "systemicmean": "map",
}

EICU_APERIODIC_COLUMNS = {
    "noninvasivesystolic": "sbp",
    "noninvasivediastolic": "dbp",
    "noninvasivemean": "map",
}

EICU_NURSE_LABELS = {
    "heart rate": "heart_rate",
    "o2 saturation": "spo2",
    "non-invasive bp systolic": "sbp",
    "non-invasive bp diastolic": "dbp",
    "non-invasive bp mean": "map",
    "invasive bp systolic": "sbp",
    "invasive bp diastolic": "dbp",
    "invasive bp mean": "map",
    "temperature (c)": "temp_c",
    "temperature (f)": "temp_f",
    "gcs total": "gcs_total",
    "eyes": "gcs_eye",
    "motor": "gcs_motor",
    "verbal": "gcs_verbal",
    "respiratory rate": "resp_rate",
    "bedside glucose": "glucose",
}

EICU_LAB_LABELS = {
    "alt (sgpt)": "alt",
    "ast (sgot)": "ast",
    "bun": "bun",
    "crp": "crp",
    "crp-hs": "crp",
    "fio2": "fio2",
    "hco3": "bicarbonate",
    "hct": "hematocrit",
    "hgb": "hemoglobin",
    "pt": "pt",
    "pt - inr": "inr",
    "ptt": "ptt",
    "wbc x 1000": "wbc",
    "bedside glucose": "glucose",
    "bicarbonate": "bicarbonate",
    "calcium": "calcium",
    "chloride": "chloride",
    "creatinine": "creatinine",
    "glucose": "glucose",
    "lactate": "lactate",
    "magnesium": "magnesium",
    "ph": "ph",
    "paco2": "paco2",
    "pao2": "pao2",
    "phosphate": "phosphate",
    "platelets x 1000": "platelet",
    "potassium": "potassium",
    "procalcitonin": "procalcitonin",
    "sodium": "sodium",
    "total bilirubin": "bilirubin_total",
}

EICU_RESP_LABELS = {
    "fio2": "fio2",
    "sao2": "spo2",
    "total rr": "resp_rate",
    "rr (patient)": "resp_rate",
    "vent rate": "resp_rate",
}


PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (15.0, 300.0), "sbp": (25.0, 350.0), "dbp": (10.0, 220.0),
    "map": (15.0, 250.0), "spo2": (20.0, 100.0), "resp_rate": (1.0, 100.0),
    "temp_c": (25.0, 45.0), "temp_f": (77.0, 113.0), "gcs_total": (3.0, 15.0),
    "gcs_verbal": (1.0, 5.0), "gcs_motor": (1.0, 6.0), "gcs_eye": (1.0, 4.0),
    "glucose": (5.0, 2000.0), "fio2": (15.0, 100.0), "urine_output": (0.0, 20000.0),
    "lactate": (0.0, 40.0), "bicarbonate": (2.0, 60.0), "ph": (6.5, 8.0),
    "pao2": (10.0, 800.0), "paco2": (5.0, 250.0), "creatinine": (0.0, 30.0),
    "bun": (0.0, 300.0), "sodium": (80.0, 220.0), "potassium": (1.0, 12.0),
    "chloride": (50.0, 180.0), "calcium": (2.0, 30.0), "magnesium": (0.2, 20.0),
    "phosphate": (0.2, 25.0), "hemoglobin": (1.0, 30.0), "hematocrit": (3.0, 80.0),
    "wbc": (0.0, 500.0), "platelet": (0.0, 2000.0), "bilirubin_total": (0.0, 100.0),
    "alt": (0.0, 50000.0), "ast": (0.0, 50000.0), "inr": (0.5, 30.0),
    "pt": (5.0, 200.0), "ptt": (5.0, 300.0), "crp": (0.0, 1000.0),
    "procalcitonin": (0.0, 1000.0),
}


ELIXHAUSER_ICD9_PREFIXES: dict[str, tuple[str, ...]] = {
    "chf": ("39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493", "4254", "4255", "4256", "4257", "4258", "4259", "428"),
    "cardiac_arrhythmia": ("4260", "42613", "4267", "4269", "42610", "42612", "4270", "4271", "4272", "4273", "4274", "4276", "4277", "4278", "4279", "7850", "99601", "99604", "V450", "V533"),
    "valvular_disease": ("0932", "394", "395", "396", "397", "424", "7463", "7464", "7465", "7466", "V422", "V433"),
    "pulmonary_circulation": ("4150", "4151", "416", "4170", "4178", "4179"),
    "pvd": ("0930", "4373", "440", "441", "4431", "4432", "4433", "4434", "4435", "4436", "4437", "4438", "4439", "4471", "5571", "5579", "V434"),
    "hypertension": ("401", "402", "403", "404", "405"),
    "paralysis": ("3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"),
    "other_neurological": ("3319", "3320", "3321", "3334", "3335", "33392", "334", "335", "3362", "340", "341", "345", "3481", "3483", "7803", "7843"),
    "copd": ("4168", "4169", "490", "491", "492", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "5064", "5081", "5088"),
    "diabetes_uncomplicated": ("2500", "2501", "2502", "2503"),
    "diabetes_complicated": ("2504", "2505", "2506", "2507", "2508", "2509"),
    "hypothyroidism": ("2409", "243", "244", "2461", "2468"),
    "renal_failure": ("40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "585", "586", "5880", "V420", "V451", "V56"),
    "liver_disease": ("07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "4560", "4561", "4562", "570", "571", "5722", "5723", "5724", "5725", "5726", "5727", "5728", "5733", "5734", "5738", "5739", "V427"),
    "peptic_ulcer": ("5317", "5319", "5327", "5329", "5337", "5339", "5347", "5349"),
    "aids": ("042", "043", "044"),
    "lymphoma": ("200", "201", "202", "2030", "2386"),
    "metastatic_cancer": ("196", "197", "198", "199"),
    "solid_tumor": ("140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195"),
    "rheumatoid_arthritis": ("446", "7010", "7100", "7101", "7102", "7103", "7104", "7108", "7109", "7112", "714", "7193", "720", "725", "7285", "72889", "72930"),
    "coagulopathy": ("286", "2871", "2873", "2874", "2875"),
    "obesity": ("2780",),
    "weight_loss": ("260", "261", "262", "263", "7832", "7994"),
    "fluid_electrolyte": ("2536", "276"),
    "blood_loss_anemia": ("2800",),
    "deficiency_anemia": ("2801", "2802", "2803", "2804", "2805", "2806", "2807", "2808", "2809", "281"),
    "alcohol_abuse": ("2652", "2911", "2912", "2913", "2915", "2916", "2917", "2918", "2919", "3030", "3039", "3050", "3575", "4255", "5353", "5710", "5711", "5712", "5713", "980", "V113"),
    "drug_abuse": ("292", "304", "3052", "3053", "3054", "3055", "3056", "3057", "3058", "3059", "V6542"),
    "psychoses": ("2938", "295", "29604", "29614", "29644", "29654", "297", "298"),
    "depression": ("2962", "2963", "2965", "3004", "309", "311"),
}

ELIXHAUSER_ICD10_PREFIXES: dict[str, tuple[str, ...]] = {
    "chf": ("I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"),
    "cardiac_arrhythmia": ("I441", "I442", "I443", "I456", "I459", "I47", "I48", "I49", "R000", "R001", "R008", "T821", "Z450", "Z950"),
    "valvular_disease": ("A520", "I05", "I06", "I07", "I08", "I091", "I098", "I34", "I35", "I36", "I37", "I38", "I39", "Q230", "Q231", "Q232", "Q233", "Z952", "Z953", "Z954"),
    "pulmonary_circulation": ("I26", "I27", "I280", "I288", "I289"),
    "pvd": ("I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"),
    "hypertension": ("I10", "I11", "I12", "I13", "I15"),
    "paralysis": ("G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"),
    "other_neurological": ("G10", "G11", "G12", "G13", "G20", "G21", "G22", "G254", "G255", "G312", "G318", "G319", "G32", "G35", "G36", "G37", "G40", "G41", "G931", "G934", "R470", "R56"),
    "copd": ("I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"),
    "diabetes_uncomplicated": ("E100", "E101", "E109", "E110", "E111", "E119", "E120", "E121", "E129", "E130", "E131", "E139", "E140", "E141", "E149"),
    "diabetes_complicated": ("E102", "E103", "E104", "E105", "E106", "E107", "E108", "E112", "E113", "E114", "E115", "E116", "E117", "E118", "E122", "E123", "E124", "E125", "E126", "E127", "E128", "E132", "E133", "E134", "E135", "E136", "E137", "E138", "E142", "E143", "E144", "E145", "E146", "E147", "E148"),
    "hypothyroidism": ("E00", "E01", "E02", "E03", "E890"),
    "renal_failure": ("I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"),
    "liver_disease": ("B18", "I85", "I864", "I982", "K70", "K711", "K713", "K714", "K715", "K717", "K72", "K73", "K74", "K760", "K762", "K763", "K764", "K765", "K766", "K767", "K768", "K769", "Z944"),
    "peptic_ulcer": ("K257", "K259", "K267", "K269", "K277", "K279", "K287", "K289"),
    "aids": ("B20", "B21", "B22", "B24"),
    "lymphoma": ("C81", "C82", "C83", "C84", "C85", "C88", "C96", "C900", "C902"),
    "metastatic_cancer": ("C77", "C78", "C79", "C80"),
    "solid_tumor": ("C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C97"),
    "rheumatoid_arthritis": ("L940", "L941", "L943", "M05", "M06", "M08", "M120", "M123", "M30", "M310", "M311", "M312", "M313", "M32", "M33", "M34", "M35", "M45", "M461", "M468", "M469"),
    "coagulopathy": ("D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695", "D696"),
    "obesity": ("E66",),
    "weight_loss": ("E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"),
    "fluid_electrolyte": ("E222", "E86", "E87"),
    "blood_loss_anemia": ("D500",),
    "deficiency_anemia": ("D508", "D509", "D51", "D52", "D53"),
    "alcohol_abuse": ("F10", "E52", "G621", "I426", "K292", "K700", "K703", "K709", "T51", "Z502", "Z714", "Z721"),
    "drug_abuse": ("F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722"),
    "psychoses": ("F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315"),
    "depression": ("F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412", "F432"),
}


def _item_source(table: str, mapping: dict[str, tuple[int, ...]], feature: str) -> str:
    values = mapping.get(feature, ())
    return f"{table}:" + "|".join(str(value) for value in values) if values else "unavailable"


def harmonization_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(DYNAMIC_FEATURES):
        display, unit, family = DYNAMIC_METADATA[feature]
        mimic_sources = [
            source for source in (
                _item_source("chartevents.itemid", MIMIC_CHART_ITEMS, feature),
                _item_source("labevents.itemid", MIMIC_LAB_ITEMS, feature),
                "outputevents.itemid:" + "|".join(map(str, MIMIC_URINE_OUTPUT_ITEMS))
                if feature == "urine_output" else "unavailable",
            ) if source != "unavailable"
        ]
        eicu_sources: list[str] = []
        for table, mapping in (
            ("vitalPeriodic", EICU_PERIODIC_COLUMNS),
            ("vitalAperiodic", EICU_APERIODIC_COLUMNS),
            ("nurseCharting", EICU_NURSE_LABELS),
            ("lab", EICU_LAB_LABELS),
            ("respiratoryCharting", EICU_RESP_LABELS),
        ):
            matches = [source for source, target in mapping.items() if target == feature]
            if matches:
                eicu_sources.append(f"{table}:" + "|".join(matches))
        if feature == "urine_output":
            eicu_sources.append("intakeOutput:urinary-output label rule")
        if feature == "gcs_total":
            eicu_sources.append("derived:gcs_eye+gcs_motor+gcs_verbal when co-observed")
        rows.append({
            "feature_index": index,
            "feature_type": "dynamic",
            "machine_name": feature,
            "display_name": display,
            "unit": unit,
            "feature_family": family,
            "mimic_iv_mapping": ";".join(mimic_sources) or "unavailable",
            "eicu_mapping": ";".join(eicu_sources) or "structurally unavailable; retain all-missing channel",
            "mimic_iii_mapping": ";".join(mimic_sources) or "unavailable",
            "normalization_policy": "active source training split only",
            "explanation_eligible": True,
        })

    unavailable_eicu = {
        "emergency_admit", "elective_admit", "insurance_medicare",
        "insurance_medicaid", "insurance_private",
    }
    for index, feature in enumerate(STATIC_FEATURES):
        if feature == "age":
            mapping = "patient demographics"
        elif feature == "gender_male":
            mapping = "patient gender"
        elif feature in ELIXHAUSER_FEATURES:
            mapping = (
                "Quan Elixhauser ICD-9/ICD-10 prefix mapping; "
                "independent domains (assign0=false)"
            )
        elif feature in {"micu", "sicu", "cicu", "nicu", "csru"}:
            mapping = "first ICU/unit type"
        else:
            mapping = "admission type or insurance"
        rows.append({
            "feature_index": index,
            "feature_type": "static",
            "machine_name": feature,
            "display_name": feature.replace("_", " ").title(),
            "unit": "years" if feature == "age" else "binary",
            "feature_family": "admission-time demographic/context",
            "mimic_iv_mapping": mapping,
            "eicu_mapping": (
                "structurally unavailable; constant zero, never target-imputed"
                if feature in unavailable_eicu else mapping
            ),
            "mimic_iii_mapping": mapping,
            "normalization_policy": "active source training split only",
            "explanation_eligible": True,
        })
    return rows
