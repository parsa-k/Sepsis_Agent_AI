"""
DuckDB interface for querying MIMIC-IV .csv.gz files directly.
All heavy lifting stays in DuckDB — never loads full tables into Python memory.
"""

import os
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIMIC_HOSP = os.path.join(BASE_DIR, "mimiciv", "3.1", "hosp")
MIMIC_ICU = os.path.join(BASE_DIR, "mimiciv", "3.1", "icu")


def _gz(folder: str, name: str) -> str:
    return os.path.join(folder, f"{name}.csv.gz")


PATHS = {
    "patients":            _gz(MIMIC_HOSP, "patients"),
    "admissions":          _gz(MIMIC_HOSP, "admissions"),
    "labevents":           _gz(MIMIC_HOSP, "labevents"),
    "d_labitems":          _gz(MIMIC_HOSP, "d_labitems"),
    "diagnoses_icd":       _gz(MIMIC_HOSP, "diagnoses_icd"),
    "d_icd_diagnoses":     _gz(MIMIC_HOSP, "d_icd_diagnoses"),
    "prescriptions":       _gz(MIMIC_HOSP, "prescriptions"),
    "microbiologyevents":  _gz(MIMIC_HOSP, "microbiologyevents"),
    "chartevents":         _gz(MIMIC_ICU, "chartevents"),
    "icustays":            _gz(MIMIC_ICU, "icustays"),
    "d_items":             _gz(MIMIC_ICU, "d_items"),
    "inputevents":         _gz(MIMIC_ICU, "inputevents"),
    "outputevents":        _gz(MIMIC_ICU, "outputevents"),
}


def get_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("SET threads TO 4")
    return conn


# ── Dashboard metrics ────────────────────────────────────────────────────────

def get_dashboard_metrics(conn: duckdb.DuckDBPyConnection) -> dict:
    total_patients = conn.execute(
        f"SELECT COUNT(DISTINCT subject_id) FROM read_csv_auto('{PATHS['patients']}')"
    ).fetchone()[0]

    total_admissions = conn.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{PATHS['admissions']}')"
    ).fetchone()[0]

    gender_df = conn.execute(
        f"SELECT gender, COUNT(*) AS cnt FROM read_csv_auto('{PATHS['patients']}') GROUP BY gender"
    ).fetchdf()

    avg_age = conn.execute(
        f"SELECT ROUND(AVG(anchor_age), 1) FROM read_csv_auto('{PATHS['patients']}')"
    ).fetchone()[0]

    admission_types = conn.execute(
        f"SELECT admission_type, COUNT(*) AS cnt FROM read_csv_auto('{PATHS['admissions']}') GROUP BY admission_type ORDER BY cnt DESC"
    ).fetchdf()

    mortality = conn.execute(
        f"SELECT SUM(hospital_expire_flag) AS deaths, COUNT(*) AS total FROM read_csv_auto('{PATHS['admissions']}')"
    ).fetchone()

    return {
        "total_patients": total_patients,
        "total_admissions": total_admissions,
        "gender_df": gender_df,
        "avg_age": avg_age,
        "admission_types": admission_types,
        "deaths": mortality[0],
        "total_for_mortality": mortality[1],
    }


def get_admission_types(conn: duckdb.DuckDBPyConnection) -> list[str]:
    df = conn.execute(
        f"SELECT DISTINCT admission_type FROM read_csv_auto('{PATHS['admissions']}') ORDER BY admission_type"
    ).fetchdf()
    return df["admission_type"].tolist()


def get_sepsis_ready_stays(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Find ICU stays that have ALL raw data required for a Sepsis-3 assessment.

    Checks 8 requirements across 6 tables (each big file scanned only once):
      Infection: blood cultures + antibiotics
      SOFA:      PaO2, Platelets, Bilirubin, Creatinine (labs)
                 MAP, GCS (chart vitals)

    Returns DataFrame with columns: stay_id, subject_id, hadm_id
    """
    return conn.execute(f"""
        WITH has_labs AS (
            SELECT hadm_id,
                MAX(CASE WHEN itemid = 50821 THEN 1 ELSE 0 END) AS has_pao2,
                MAX(CASE WHEN itemid = 51265 THEN 1 ELSE 0 END) AS has_platelets,
                MAX(CASE WHEN itemid = 50885 THEN 1 ELSE 0 END) AS has_bilirubin,
                MAX(CASE WHEN itemid = 50912 THEN 1 ELSE 0 END) AS has_creatinine
            FROM read_csv_auto('{PATHS['labevents']}')
            WHERE itemid IN (50821, 51265, 50885, 50912)
            GROUP BY hadm_id
            HAVING has_pao2 = 1
               AND has_platelets = 1
               AND has_bilirubin = 1
               AND has_creatinine = 1
        ),
        has_vitals AS (
            SELECT stay_id,
                MAX(CASE WHEN itemid IN (220052, 220181) THEN 1 ELSE 0 END) AS has_map,
                MAX(CASE WHEN itemid IN (220739, 223900, 223901) THEN 1 ELSE 0 END) AS has_gcs
            FROM read_csv_auto('{PATHS['chartevents']}')
            WHERE itemid IN (220052, 220181, 220739, 223900, 223901)
            GROUP BY stay_id
            HAVING has_map = 1 AND has_gcs = 1
        ),
        has_cultures AS (
            SELECT DISTINCT hadm_id
            FROM read_csv_auto('{PATHS['microbiologyevents']}')
            WHERE hadm_id IS NOT NULL
        ),
        has_antibiotics AS (
            SELECT DISTINCT hadm_id
            FROM read_csv_auto('{PATHS['prescriptions']}')
            WHERE drug_type IN ('MAIN', 'BASE')
              AND regexp_matches(
                lower(drug),
                '(cillin|mycin|cycline|floxacin|penem|cef|sulfa|vanco|metro'
                '|azole|oxacin|clinda|dapto|linezol|rifamp|nitrofur|trimeth|fosfo'
                '|tobra|genta|amika|strepto|colistin|polymyxin|tigecycl)'
              )
        )
        SELECT DISTINCT icu.stay_id, icu.subject_id, icu.hadm_id
        FROM read_csv_auto('{PATHS['icustays']}') icu
        INNER JOIN has_labs ON icu.hadm_id = has_labs.hadm_id
        INNER JOIN has_vitals ON icu.stay_id = has_vitals.stay_id
        INNER JOIN has_cultures ON icu.hadm_id = has_cultures.hadm_id
        INNER JOIN has_antibiotics ON icu.hadm_id = has_antibiotics.hadm_id
    """).fetchdf()


def get_patients_filtered(
    conn: duckdb.DuckDBPyConnection,
    offset: int,
    limit: int,
    has_icu: str = "All",
    admission_type: str = "All",
    sepsis_ready_subjects: set[int] | None = None,
) -> tuple[pd.DataFrame, int]:
    icu_join = f"LEFT JOIN read_csv_auto('{PATHS['icustays']}') icu ON p.subject_id = icu.subject_id AND a.hadm_id = icu.hadm_id"

    where_clauses = ["1=1"]
    if admission_type != "All":
        where_clauses.append(f"a.admission_type = '{admission_type}'")
    if sepsis_ready_subjects is not None:
        id_list = ",".join(str(s) for s in sepsis_ready_subjects)
        where_clauses.append(f"p.subject_id IN ({id_list})")

    having_clause = ""
    if has_icu == "ICU patients only":
        having_clause = "HAVING COUNT(DISTINCT icu.stay_id) > 0"
    elif has_icu == "Non-ICU patients only":
        having_clause = "HAVING COUNT(DISTINCT icu.stay_id) = 0"

    where_sql = " AND ".join(where_clauses)

    count = conn.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT p.subject_id
            FROM read_csv_auto('{PATHS['patients']}') p
            JOIN read_csv_auto('{PATHS['admissions']}') a USING (subject_id)
            {icu_join}
            WHERE {where_sql}
            GROUP BY p.subject_id
            {having_clause}
        ) sub
    """).fetchone()[0]

    df = conn.execute(f"""
        SELECT p.subject_id, p.gender, p.anchor_age, p.anchor_year_group, p.dod,
               COUNT(DISTINCT a.hadm_id) AS num_admissions,
               COUNT(DISTINCT icu.stay_id) AS icu_stays,
               MAX(a.admission_type) AS last_admission_type
        FROM read_csv_auto('{PATHS['patients']}') p
        JOIN read_csv_auto('{PATHS['admissions']}') a USING (subject_id)
        {icu_join}
        WHERE {where_sql}
        GROUP BY p.subject_id, p.gender, p.anchor_age, p.anchor_year_group, p.dod
        {having_clause}
        ORDER BY p.subject_id
        LIMIT {limit} OFFSET {offset}
    """).fetchdf()

    return df, count


def get_patient_full_detail(conn: duckdb.DuckDBPyConnection, subject_id: int) -> dict:
    demo = conn.execute(f"""
        SELECT * FROM read_csv_auto('{PATHS['patients']}')
        WHERE subject_id = {subject_id}
    """).fetchdf()

    admissions = conn.execute(f"""
        SELECT * FROM read_csv_auto('{PATHS['admissions']}')
        WHERE subject_id = {subject_id}
        ORDER BY admittime DESC
    """).fetchdf()

    icu = conn.execute(f"""
        SELECT * FROM read_csv_auto('{PATHS['icustays']}')
        WHERE subject_id = {subject_id}
        ORDER BY intime DESC
    """).fetchdf()

    diagnoses = conn.execute(f"""
        SELECT d.hadm_id, d.seq_num, d.icd_code, d.icd_version, i.long_title
        FROM read_csv_auto('{PATHS['diagnoses_icd']}') d
        JOIN read_csv_auto('{PATHS['d_icd_diagnoses']}') i
          ON d.icd_code = i.icd_code AND d.icd_version = i.icd_version
        WHERE d.subject_id = {subject_id}
        ORDER BY d.hadm_id, d.seq_num
    """).fetchdf()

    labs = conn.execute(f"""
        SELECT l.hadm_id, l.charttime, l.value, l.valuenum, l.valueuom, l.flag,
               d.label
        FROM read_csv_auto('{PATHS['labevents']}') l
        JOIN read_csv_auto('{PATHS['d_labitems']}') d ON l.itemid = d.itemid
        WHERE l.subject_id = {subject_id}
        ORDER BY l.hadm_id, l.charttime DESC
    """).fetchdf()

    vitals = conn.execute(f"""
        SELECT c.hadm_id, c.stay_id, c.charttime, c.valuenum, c.valueuom,
               d.label
        FROM read_csv_auto('{PATHS['chartevents']}') c
        JOIN read_csv_auto('{PATHS['d_items']}') d ON c.itemid = d.itemid
        WHERE c.subject_id = {subject_id}
        ORDER BY c.hadm_id, c.charttime DESC
    """).fetchdf()

    prescriptions = conn.execute(f"""
        SELECT hadm_id, starttime, stoptime, drug, drug_type, route,
               dose_val_rx, dose_unit_rx
        FROM read_csv_auto('{PATHS['prescriptions']}')
        WHERE subject_id = {subject_id}
        ORDER BY hadm_id, starttime DESC
    """).fetchdf()

    micro = conn.execute(f"""
        SELECT hadm_id, chartdate, charttime, spec_type_desc, test_name,
               org_name, ab_name, interpretation
        FROM read_csv_auto('{PATHS['microbiologyevents']}')
        WHERE subject_id = {subject_id}
        ORDER BY hadm_id, chartdate DESC
    """).fetchdf()

    inputs = conn.execute(f"""
        SELECT i.hadm_id, i.stay_id, i.starttime, i.endtime, d.label,
               i.amount, i.amountuom, i.rate, i.rateuom, i.ordercategoryname
        FROM read_csv_auto('{PATHS['inputevents']}') i
        JOIN read_csv_auto('{PATHS['d_items']}') d ON i.itemid = d.itemid
        WHERE i.subject_id = {subject_id}
        ORDER BY i.hadm_id, i.starttime DESC
    """).fetchdf()

    return {
        "demographics": demo,
        "admissions": admissions,
        "icu_stays": icu,
        "diagnoses": diagnoses,
        "labs": labs,
        "vitals": vitals,
        "prescriptions": prescriptions,
        "microbiology": micro,
        "input_events": inputs,
    }


# ── Agent workspace queries ──────────────────────────────────────────────────

def find_patient(conn: duckdb.DuckDBPyConnection, subject_id: int = None, hadm_id: int = None) -> pd.DataFrame:
    if hadm_id:
        return conn.execute(f"""
            SELECT a.*, p.gender, p.anchor_age, p.dod
            FROM read_csv_auto('{PATHS['admissions']}') a
            JOIN read_csv_auto('{PATHS['patients']}') p USING (subject_id)
            WHERE a.hadm_id = {hadm_id}
        """).fetchdf()
    return conn.execute(f"""
        SELECT a.*, p.gender, p.anchor_age, p.dod
        FROM read_csv_auto('{PATHS['admissions']}') a
        JOIN read_csv_auto('{PATHS['patients']}') p USING (subject_id)
        WHERE a.subject_id = {subject_id}
        ORDER BY a.admittime DESC
        LIMIT 1
    """).fetchdf()


def get_vitals(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Fetch ICU charted vitals (HR, BP, Temp, RR, SpO2, GCS)."""
    vital_items = {
        220045: "Heart Rate",
        220050: "Arterial BP Systolic",
        220051: "Arterial BP Diastolic",
        220052: "Arterial BP Mean",
        220179: "Non Invasive BP Systolic",
        220180: "Non Invasive BP Diastolic",
        220181: "Non Invasive BP Mean",
        223761: "Temperature F",
        223762: "Temperature C",
        220210: "Respiratory Rate",
        220277: "SpO2",
        220739: "GCS - Eye Opening",
        223900: "GCS - Verbal Response",
        223901: "GCS - Motor Response",
        226104: "GCS Total",  # not always charted
    }
    item_ids = ",".join(str(i) for i in vital_items)
    df = conn.execute(f"""
        SELECT c.charttime, c.itemid, c.value, c.valuenum, c.valueuom,
               d.label
        FROM read_csv_auto('{PATHS['chartevents']}') c
        JOIN read_csv_auto('{PATHS['d_items']}') d ON c.itemid = d.itemid
        WHERE c.subject_id = {subject_id}
          AND c.hadm_id = {hadm_id}
          AND c.itemid IN ({item_ids})
        ORDER BY c.charttime
    """).fetchdf()
    return df


def get_labs(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Fetch labs for an admission — includes WBC, Lactate, Creatinine, Bilirubin, Platelets, etc."""
    sepsis_labs = {
        51301: "WBC",
        51222: "Hemoglobin",
        51265: "Platelets",
        50912: "Creatinine",
        50885: "Bilirubin Total",
        50813: "Lactate",
        50818: "pCO2",
        50821: "pO2",
        50820: "pH",
        51003: "Troponin T",
        51006: "Urea Nitrogen (BUN)",
        50931: "Glucose",
        50983: "Sodium",
        50971: "Potassium",
        50882: "Bicarbonate",
        51144: "Bands",
    }
    item_ids = ",".join(str(i) for i in sepsis_labs)
    df = conn.execute(f"""
        SELECT l.charttime, l.itemid, l.value, l.valuenum, l.valueuom, l.flag,
               d.label
        FROM read_csv_auto('{PATHS['labevents']}') l
        JOIN read_csv_auto('{PATHS['d_labitems']}') d ON l.itemid = d.itemid
        WHERE l.subject_id = {subject_id}
          AND l.hadm_id = {hadm_id}
          AND l.itemid IN ({item_ids})
        ORDER BY l.charttime
    """).fetchdf()
    return df


def get_microbiology(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT chartdate, charttime, spec_type_desc, test_name, org_name, ab_name,
               interpretation
        FROM read_csv_auto('{PATHS['microbiologyevents']}')
        WHERE subject_id = {subject_id}
          AND hadm_id = {hadm_id}
        ORDER BY chartdate
    """).fetchdf()


def get_prescriptions(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT starttime, stoptime, drug, drug_type, route, dose_val_rx, dose_unit_rx
        FROM read_csv_auto('{PATHS['prescriptions']}')
        WHERE subject_id = {subject_id}
          AND hadm_id = {hadm_id}
        ORDER BY starttime
    """).fetchdf()


def get_icu_stays(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT * FROM read_csv_auto('{PATHS['icustays']}')
        WHERE subject_id = {subject_id}
          AND hadm_id = {hadm_id}
        ORDER BY intime
    """).fetchdf()


def get_diagnoses(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int = None) -> pd.DataFrame:
    where = f"d.subject_id = {subject_id}"
    if hadm_id:
        where += f" AND d.hadm_id = {hadm_id}"
    return conn.execute(f"""
        SELECT d.hadm_id, d.seq_num, d.icd_code, d.icd_version, i.long_title
        FROM read_csv_auto('{PATHS['diagnoses_icd']}') d
        JOIN read_csv_auto('{PATHS['d_icd_diagnoses']}') i
          ON d.icd_code = i.icd_code AND d.icd_version = i.icd_version
        WHERE {where}
        ORDER BY d.hadm_id, d.seq_num
    """).fetchdf()


def get_historical_admissions(conn: duckdb.DuckDBPyConnection, subject_id: int, current_hadm_id: int) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT * FROM read_csv_auto('{PATHS['admissions']}')
        WHERE subject_id = {subject_id}
          AND hadm_id != {current_hadm_id}
        ORDER BY admittime
    """).fetchdf()


def get_input_events(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT i.starttime, i.endtime, i.itemid, d.label, i.amount, i.amountuom,
               i.rate, i.rateuom, i.ordercategoryname
        FROM read_csv_auto('{PATHS['inputevents']}') i
        JOIN read_csv_auto('{PATHS['d_items']}') d ON i.itemid = d.itemid
        WHERE i.subject_id = {subject_id}
          AND i.hadm_id = {hadm_id}
        ORDER BY i.starttime
    """).fetchdf()


def get_output_events(conn: duckdb.DuckDBPyConnection, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Fetch output events (urine, drains, etc.) for an admission."""
    return conn.execute(f"""
        SELECT o.charttime, o.itemid, d.label, o.value, o.valueuom
        FROM read_csv_auto('{PATHS['outputevents']}') o
        JOIN read_csv_auto('{PATHS['d_items']}') d ON o.itemid = d.itemid
        WHERE o.subject_id = {subject_id}
          AND o.hadm_id = {hadm_id}
        ORDER BY o.charttime
    """).fetchdf()
