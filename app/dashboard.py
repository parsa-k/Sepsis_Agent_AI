"""MIMIC-IV Dataset Dashboard page."""

import streamlit as st
import plotly.express as px

import db


def render():
    st.markdown("# 📊 MIMIC-IV Dataset Dashboard")
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    metrics = _load_dashboard()

    _render_metric_cards(metrics)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_charts(metrics)
    _render_patient_browser()


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner="Querying MIMIC-IV...")
def _load_dashboard():
    conn = db.get_conn()
    metrics = db.get_dashboard_metrics(conn)
    conn.close()
    return metrics


@st.cache_data(ttl=600, show_spinner="Loading filter options...")
def _load_admission_types():
    conn = db.get_conn()
    types = db.get_admission_types(conn)
    conn.close()
    return types


@st.cache_data(
    ttl=3600,
    show_spinner=(
        "Identifying sepsis-ready patients (scanning labs, vitals, "
        "cultures, antibiotics)... This may take a few minutes on first run."
    ),
)
def _load_sepsis_ready():
    conn = db.get_conn()
    df = db.get_sepsis_ready_stays(conn)
    conn.close()
    return df


@st.cache_data(ttl=300, show_spinner="Querying patients...")
def _load_filtered_page(
    offset, limit, has_icu, admission_type,
    _sepsis_ids_hash, sepsis_ids_tuple,
):
    sr = set(sepsis_ids_tuple) if sepsis_ids_tuple else None
    conn = db.get_conn()
    df, count = db.get_patients_filtered(
        conn, offset, limit, has_icu, admission_type,
        sepsis_ready_subjects=sr,
    )
    conn.close()
    return df, count


@st.cache_data(ttl=300, show_spinner="Loading full patient record...")
def _load_full_detail(sid):
    conn = db.get_conn()
    detail = db.get_patient_full_detail(conn, sid)
    conn.close()
    return detail


# ── Metric cards ─────────────────────────────────────────────────────────────

def _render_metric_cards(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics["total_patients"]:,}</div>'
            f'<div class="metric-label">Total Patients</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card metric-card-blue">'
            f'<div class="metric-value">{metrics["total_admissions"]:,}</div>'
            f'<div class="metric-label">Total Admissions</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card metric-card-green">'
            f'<div class="metric-value">{metrics["avg_age"]}</div>'
            f'<div class="metric-label">Average Age</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        mort_rate = (
            round(metrics["deaths"] / metrics["total_for_mortality"] * 100, 1)
            if metrics["total_for_mortality"] else 0
        )
        st.markdown(
            f'<div class="metric-card metric-card-red">'
            f'<div class="metric-value">{mort_rate}%</div>'
            f'<div class="metric-label">Mortality Rate</div></div>',
            unsafe_allow_html=True,
        )


# ── Charts ───────────────────────────────────────────────────────────────────

def _render_charts(metrics: dict):
    chart1, chart2 = st.columns(2)

    with chart1:
        gender_df = metrics["gender_df"]
        fig = px.pie(
            gender_df, names="gender", values="cnt",
            title="Gender Distribution",
            color_discrete_sequence=["#667eea", "#f45c43", "#38ef7d"],
            hole=0.45,
        )
        fig.update_layout(
            font_family="Inter",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        adm_df = metrics["admission_types"].head(8)
        fig = px.bar(
            adm_df, x="cnt", y="admission_type",
            orientation="h", title="Top Admission Types",
            color="cnt",
            color_continuous_scale=["#667eea", "#764ba2"],
        )
        fig.update_layout(
            font_family="Inter",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis_title="", xaxis_title="Count",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Patient Browser ──────────────────────────────────────────────────────────

PAGE_SIZE = 25


def _render_patient_browser():
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Patient Browser")

    admission_types_list = _load_admission_types()

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 0.7])
    with f1:
        icu_filter = st.selectbox(
            "Patient Type",
            ["All", "ICU patients only", "Non-ICU patients only"],
            key="icu_filter",
        )
    with f2:
        adm_filter = st.selectbox(
            "Admission Type",
            ["All"] + admission_types_list,
            key="adm_type_filter",
        )
    with f3:
        sepsis_filter = st.selectbox(
            "Sepsis Data",
            ["All", "Sepsis-ready only"],
            key="sepsis_data_filter",
            help=(
                "Filter to patients with ALL data required for Sepsis-3 "
                "diagnosis: blood cultures, antibiotics, PaO2, Platelets, "
                "Bilirubin, Creatinine, MAP, and GCS."
            ),
        )
    with f4:
        current_page = st.number_input(
            "Page", min_value=1, value=1, step=1, key="browser_page",
        )

    sepsis_subjects = None
    if sepsis_filter == "Sepsis-ready only":
        sepsis_df = _load_sepsis_ready()
        sepsis_subjects = set(sepsis_df["subject_id"].unique().tolist())
        sepsis_stays_count = len(sepsis_df)
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#667eea,#764ba2);'
            f'color:white;border-radius:10px;padding:0.6rem 1.2rem;'
            f'margin-bottom:0.75rem;font-size:0.85rem;">'
            f'Sepsis-ready: <strong>{len(sepsis_subjects):,}</strong> '
            f'patients across <strong>{sepsis_stays_count:,}</strong> ICU '
            f'stays have complete data (cultures + antibiotics + PaO2 + '
            f'Platelets + Bilirubin + Creatinine + MAP + GCS)</div>',
            unsafe_allow_html=True,
        )

    sepsis_ids_tuple = (
        tuple(sorted(sepsis_subjects)) if sepsis_subjects else None
    )
    sepsis_hash = hash(sepsis_ids_tuple) if sepsis_ids_tuple else 0

    offset = (current_page - 1) * PAGE_SIZE
    patients_df, total_count = _load_filtered_page(
        offset, PAGE_SIZE, icu_filter, adm_filter,
        sepsis_hash, sepsis_ids_tuple,
    )
    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)

    st.markdown(
        f"<small style='color:#888;'>Showing page {current_page} of "
        f"{total_pages:,} ({total_count:,} patients match filters)</small>",
        unsafe_allow_html=True,
    )

    event = st.dataframe(
        patients_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        sel_subject_id = int(patients_df.iloc[selected_rows[0]]["subject_id"])
        _render_patient_detail(sel_subject_id)


# ── Patient detail drill-down ────────────────────────────────────────────────

_SECTIONS = [
    ("Demographics",   "demographics",   True,  True),
    ("Admissions",     "admissions",     True,  True),
    ("ICU Stays",      "icu_stays",      False, False),
    ("Diagnoses — ICD Codes", "diagnoses", False, False),
    ("Laboratory Results",    "labs",      False, False),
    ("Vitals — Chart Events", "vitals",   False, False),
    ("Prescriptions",  "prescriptions",  False, False),
    ("Microbiology Cultures", "microbiology", False, False),
    ("Input Events — IV Fluids & Infusions", "input_events", False, False),
]


def _render_patient_detail(subject_id: int):
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"### Complete Patient Record — `subject_id = {subject_id}`"
    )

    detail = _load_full_detail(subject_id)
    available, missing = [], []

    for label, key, expanded, _ in _SECTIONS:
        df = detail[key]
        count_str = f" ({len(df)} records)" if key != "demographics" else ""
        with st.expander(f"**{label}**{count_str}", expanded=expanded):
            if not df.empty:
                display = df.head(500)
                st.dataframe(
                    display, use_container_width=True, hide_index=True,
                )
                if len(df) > 500:
                    st.caption(
                        f"Showing first 500 of {len(df):,} records."
                    )
                available.append(label)
            else:
                st.info(f"No {label.lower()} found.")
                missing.append(label)

    _render_completeness_summary(available, missing)


def _render_completeness_summary(
    available: list[str], missing: list[str],
):
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Data Completeness Summary")

    total = len(available) + len(missing)
    sum1, sum2 = st.columns(2)

    with sum1:
        items = "".join(
            f'<div style="color:#333;">&#10003; {s}</div>'
            for s in available
        ) or '<div style="color:#888;">None</div>'
        st.markdown(
            f'<div style="background:#e8f5e9;border-radius:12px;'
            f'padding:1rem 1.5rem;border-left:4px solid #4caf50;">'
            f'<div style="font-weight:600;color:#2e7d32;'
            f'margin-bottom:0.5rem;">Available Data '
            f'({len(available)} / {total})</div>{items}</div>',
            unsafe_allow_html=True,
        )

    with sum2:
        if missing:
            items = "".join(
                f'<div style="color:#333;">&#10007; {s}</div>'
                for s in missing
            )
            st.markdown(
                f'<div style="background:#fff3e0;border-radius:12px;'
                f'padding:1rem 1.5rem;border-left:4px solid #ff9800;">'
                f'<div style="font-weight:600;color:#e65100;'
                f'margin-bottom:0.5rem;">Missing / Empty Data '
                f'({len(missing)})</div>{items}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#e8f5e9;border-radius:12px;'
                'padding:1rem 1.5rem;border-left:4px solid #4caf50;">'
                '<div style="font-weight:600;color:#2e7d32;">'
                'All data sections are present for this patient.'
                '</div></div>',
                unsafe_allow_html=True,
            )
