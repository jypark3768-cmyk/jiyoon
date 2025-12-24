import io
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# Page / Font
# -----------------------------
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"


# -----------------------------
# Constants (학교/EC 조건은 연구 설계값이므로 상수로 둡니다)
# 파일명/시트명 하드코딩은 하지 않습니다.
# -----------------------------
EC_TARGET_BY_SCHOOL: Dict[str, float] = {
    "송도고": 1.0,
    "하늘고": 2.0,  # (최적 후보)
    "아라고": 4.0,
    "동산고": 8.0,
}

SCHOOL_COLOR: Dict[str, str] = {
    "송도고": "Blue",
    "하늘고": "Green",
    "아라고": "Orange",
    "동산고": "Red",
}

ENV_REQUIRED_COLS = ["time", "temperature", "humidity", "ph", "ec"]


# -----------------------------
# Unicode-safe helpers (NFC/NFD)
# -----------------------------
def _norm_variants(text: str) -> Tuple[str, str]:
    """Return (NFC, NFD) variants."""
    return (unicodedata.normalize("NFC", text), unicodedata.normalize("NFD", text))


def _norm_eq(a: str, b: str) -> bool:
    """Bidirectional NFC/NFD equivalence check."""
    a_nfc, a_nfd = _norm_variants(a)
    b_nfc, b_nfd = _norm_variants(b)
    return (a_nfc == b_nfc) or (a_nfc == b_nfd) or (a_nfd == b_nfc) or (a_nfd == b_nfd)


def _contains_norm(haystack: str, needle: str) -> bool:
    """NFC/NFD bidirectional substring check."""
    h_nfc, h_nfd = _norm_variants(haystack)
    n_nfc, n_nfd = _norm_variants(needle)
    return (n_nfc in h_nfc) or (n_nfd in h_nfc) or (n_nfc in h_nfd) or (n_nfd in h_nfd)


def _lookup_by_norm_key(mapping: Dict[str, float], key: str) -> Optional[float]:
    for k, v in mapping.items():
        if _norm_eq(k, key):
            return v
    return None


# -----------------------------
# File discovery (NO glob-only, use Path.iterdir)
# -----------------------------
def discover_data_files(data_dir: Path) -> Tuple[Tuple[str, ...], Optional[str]]:
    """
    Returns:
      - csv_paths: tuple of CSV file paths (환경 데이터 후보)
      - xlsx_path: one XLSX path (생육 결과 후보), or None
    Rules:
      - Use Path.iterdir()
      - Unicode normalize checks (NFC/NFD) for selecting best XLSX if multiple exist
    """
    if not data_dir.exists() or not data_dir.is_dir():
        return tuple(), None

    csv_paths: List[str] = []
    xlsx_candidates: List[Path] = []

    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            csv_paths.append(str(p))
        elif suf == ".xlsx":
            xlsx_candidates.append(p)

    chosen_xlsx: Optional[Path] = None
    if len(xlsx_candidates) == 1:
        chosen_xlsx = xlsx_candidates[0]
    elif len(xlsx_candidates) > 1:
        # Prefer a file whose name contains '생육' or '결과' (NFC/NFD-safe), else take the first.
        preferred: List[Path] = []
        for p in xlsx_candidates:
            name = p.name
            if _contains_norm(name, "생육") or _contains_norm(name, "결과"):
                preferred.append(p)
        chosen_xlsx = preferred[0] if len(preferred) > 0 else xlsx_candidates[0]

    return tuple(csv_paths), (str(chosen_xlsx) if chosen_xlsx is not None else None)


# -----------------------------
# Robust readers
# -----------------------------
def _read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError("CSV 읽기 실패: {}".format(path.name)) from last_err


def _standardize_env_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols
    missing = [c for c in ENV_REQUIRED_COLS if c not in df.columns]
    if len(missing) > 0:
        raise ValueError("환경 CSV 필수 컬럼 누락: {}".format(", ".join(missing)))
    return df


def _find_col_by_keywords(cols: Iterable[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        ok = True
        for k in keywords:
            if k not in c:
                ok = False
                break
        if ok:
            return c
    return None


def _standardize_growth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    목표 내부 컬럼:
      - 개체번호
      - 잎 수(장)
      - 지상부 길이(mm)
      - 지하부길이(mm)
      - 생중량(g)
    시트마다 공백/약간의 표기 차이를 대비해 키워드 기반 매칭.
    """
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    cols = list(df2.columns)

    # Try exact first, then keyword match
    need = {
        "개체번호": ["개체", "번호"],
        "잎 수(장)": ["잎", "수"],
        "지상부 길이(mm)": ["지상부", "길이"],
        "지하부길이(mm)": ["지하부", "길이"],
        "생중량(g)": ["생", "중량"],
    }

    rename_map: Dict[str, str] = {}

    for target, keys in need.items():
        if target in cols:
            continue
        found = _find_col_by_keywords(cols, keys)
        if found is not None:
            rename_map[found] = target

    if len(rename_map) > 0:
        df2 = df2.rename(columns=rename_map)

    missing = [t for t in need.keys() if t not in df2.columns]
    if len(missing) > 0:
        raise ValueError("생육 데이터 필수 컬럼 누락: {}".format(", ".join(missing)))

    return df2


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_environment_data(csv_paths: Tuple[str, ...]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p_str in csv_paths:
        p = Path(p_str)
        df = _read_csv_safely(p)
        df = _standardize_env_columns(df)

        # 학교명: 파일명에서 '_' 앞부분을 사용 (파일명 하드코딩 X)
        stem = p.stem
        parts = stem.split("_")
        school_raw = parts[0] if len(parts) > 0 else stem
        school_nfc = unicodedata.normalize("NFC", school_raw)

        df["학교"] = school_nfc

        # time parse
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        for c in ["temperature", "humidity", "ph", "ec"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["time"])
        frames.append(df)

    if len(frames) == 0:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["학교", "time"])
    return out


@st.cache_data(show_spinner=False)
def load_growth_data(xlsx_path: str) -> pd.DataFrame:
    p = Path(xlsx_path)
    if not p.exists():
        return pd.DataFrame()

    xls = pd.ExcelFile(p, engine="openpyxl")
    sheets = list(xls.sheet_names)  # 시트명 하드코딩 금지

    frames: List[pd.DataFrame] = []
    for sh in sheets:
        df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
        df = _standardize_growth_columns(df)

        school_nfc = unicodedata.normalize("NFC", str(sh).strip())
        df["학교"] = school_nfc

        # numeric
        for c in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(df)

    if len(frames) == 0:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # EC 목표 붙이기 (학교명 NFC/NFD 안전 비교)
    out["EC 목표"] = out["학교"].apply(lambda s: _lookup_by_norm_key(EC_TARGET_BY_SCHOOL, s))
    return out


# -----------------------------
# Download helpers (BytesIO)
# -----------------------------
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    # Excel 호환을 위해 utf-8-sig
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes_by_school(df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if "학교" in df.columns and df["학교"].nunique() > 1:
            for school, sdf in df.groupby("학교", dropna=False):
                sheet_name = str(school)[:31] if pd.notna(school) else "Unknown"
                sdf.to_excel(writer, index=False, sheet_name=sheet_name)
        else:
            sheet_name = "데이터"
            if "학교" in df.columns and df["학교"].nunique() == 1:
                sheet_name = str(df["학교"].iloc[0])[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    buffer.seek(0)
    return buffer


# -----------------------------
# Compute summaries
# -----------------------------
def env_means_by_school(env_df: pd.DataFrame) -> pd.DataFrame:
    if env_df.empty:
        return pd.DataFrame()
    g = env_df.groupby("학교", as_index=False).agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        ph=("ph", "mean"),
        ec=("ec", "mean"),
        n=("ec", "size"),
    )
    return g


def growth_means_by_ec(growth_df: pd.DataFrame) -> pd.DataFrame:
    if growth_df.empty:
        return pd.DataFrame()
    g = growth_df.groupby("EC 목표", as_index=False).agg(
        mean_weight=("생중량(g)", "mean"),
        mean_leaves=("잎 수(장)", "mean"),
        mean_shoot=("지상부 길이(mm)", "mean"),
        count=("생중량(g)", "count"),
    )
    g = g.sort_values("EC 목표")
    return g


def best_ec_from_growth(growth_df: pd.DataFrame) -> Optional[float]:
    g = growth_means_by_ec(growth_df)
    if g.empty or g["mean_weight"].dropna().empty:
        return None
    best_row = g.loc[g["mean_weight"].idxmax()]
    val = best_row["EC 목표"]
    return float(val) if pd.notna(val) else None


# -----------------------------
# Load data
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

with st.spinner("데이터 파일을 불러오는 중..."):
    csv_paths, xlsx_path = discover_data_files(DATA_DIR)

if len(csv_paths) == 0:
    st.error("data/ 폴더에서 환경 데이터 CSV 파일을 찾지 못했습니다. (확장자 .csv)")
    st.stop()

if xlsx_path is None:
    st.error("data/ 폴더에서 생육 결과 XLSX 파일을 찾지 못했습니다. (확장자 .xlsx)")
    st.stop()

try:
    with st.spinner("환경 데이터를 읽는 중..."):
        env_df = load_environment_data(csv_paths)
    with st.spinner("생육 결과 데이터를 읽는 중..."):
        growth_df = load_growth_data(xlsx_path)
except Exception as e:
    st.error("데이터 로딩 중 오류가 발생했습니다: {}".format(e))
    st.stop()

if env_df.empty:
    st.error("환경 데이터가 비어 있습니다. CSV 내용을 확인해주세요.")
    st.stop()

if growth_df.empty:
    st.error("생육 결과 데이터가 비어 있습니다. XLSX/시트 내용을 확인해주세요.")
    st.stop()

# 학교 목록(환경+생육 합집합)
schools_env = sorted(list({unicodedata.normalize("NFC", s) for s in env_df["학교"].dropna().unique()}))
schools_growth = sorted(list({unicodedata.normalize("NFC", s) for s in growth_df["학교"].dropna().unique()}))
schools_all = sorted(list(set(schools_env).union(set(schools_growth))))

# Sidebar
st.title("🌱 극지식물 최적 EC 농도 연구")

selected_school = st.sidebar.selectbox(
    "학교 선택",
    options=["전체"] + schools_all,
    index=0,
)

# Filtered views for KPI / raw tables
if selected_school == "전체":
    env_scope = env_df.copy()
    growth_scope = growth_df.copy()
else:
    env_scope = env_df[env_df["학교"].apply(lambda x: _norm_eq(str(x), selected_school))].copy()
    growth_scope = growth_df[growth_df["학교"].apply(lambda x: _norm_eq(str(x), selected_school))].copy()


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.subheader("연구 배경 및 목적")
    st.markdown(
        """
- 극지식물(저온·저광 환경 적응 식물)의 안정적인 생장을 위해 **양액의 EC(전기전도도) 농도 최적화**가 중요합니다.
- 본 연구는 4개 학교가 서로 다른 EC 조건(1.0 / 2.0 / 4.0 / 8.0)에서 재배하며,
  **환경(온도·습도·pH·EC)** 및 **생육 결과(생중량·잎 수·길이)**를 비교하여 **최적 EC**를 도출합니다.
"""
    )

    # 학교별 EC 조건 표
    counts_by_school = (
        growth_df.groupby("학교", as_index=False)
        .agg(개체수=("생중량(g)", "count"))
        .sort_values("학교")
    )
    table_rows: List[Dict[str, object]] = []
    for _, r in counts_by_school.iterrows():
        sch = str(r["학교"])
        ec_t = _lookup_by_norm_key(EC_TARGET_BY_SCHOOL, sch)
        color = "Gray"
        for k, v in SCHOOL_COLOR.items():
            if _norm_eq(k, sch):
                color = v
                break

        table_rows.append(
            {
                "학교명": sch,
                "EC 목표": ec_t,
                "개체수": int(r["개체수"]),
                "색상": color,
            }
        )
    cond_df = pd.DataFrame(table_rows)

    st.markdown("#### 학교별 EC 조건")
    st.dataframe(cond_df, use_container_width=True, hide_index=True)

    # KPI cards
    total_n = int(growth_scope["생중량(g)"].count()) if not growth_scope.empty else 0
    avg_temp = float(env_scope["temperature"].mean()) if not env_scope.empty else float("nan")
    avg_hum = float(env_scope["humidity"].mean()) if not env_scope.empty else float("nan")
    best_ec = best_ec_from_growth(growth_df)  # 최적 EC는 전체 데이터 기준으로 도출

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", "{}".format(total_n))
    c2.metric("평균 온도(°C)", "-" if pd.isna(avg_temp) else "{:.2f}".format(avg_temp))
    c3.metric("평균 습도(%)", "-" if pd.isna(avg_hum) else "{:.2f}".format(avg_hum))
    c4.metric("최적 EC(전체 기준)", "-" if best_ec is None else "{:.1f}".format(best_ec))

    # 안내
    if selected_school != "전체":
        st.caption("현재 선택: {}  |  최적 EC는 전체(4개교) 생육 결과를 기준으로 산출합니다.".format(selected_school))


# -----------------------------
# Tab 2: Environment
# -----------------------------
with tab2:
    st.subheader("학교별 환경 평균 비교")

    env_mean = env_means_by_school(env_df)
    if env_mean.empty:
        st.error("환경 평균을 계산할 수 없습니다. 데이터/컬럼을 확인해주세요.")
        st.stop()

    # 정렬 기준: schools_all 순서 유지
    env_mean["학교_sort"] = env_mean["학교"].apply(lambda s: schools_all.index(s) if s in schools_all else 9999)
    env_mean = env_mean.sort_values("학교_sort").drop(columns=["학교_sort"])

    schools_x = env_mean["학교"].tolist()
    temp_y = env_mean["temperature"].tolist()
    hum_y = env_mean["humidity"].tolist()
    ph_y = env_mean["ph"].tolist()
    ec_measured_y = env_mean["ec"].tolist()

    # target EC list aligned
    ec_target_y: List[Optional[float]] = []
    for sch in schools_x:
        ec_target_y.append(_lookup_by_norm_key(EC_TARGET_BY_SCHOOL, sch))

    fig_env = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC(평균)"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig_env.add_trace(go.Bar(x=schools_x, y=temp_y, name="평균 온도"), row=1, col=1)
    fig_env.add_trace(go.Bar(x=schools_x, y=hum_y, name="평균 습도"), row=1, col=2)
    fig_env.add_trace(go.Bar(x=schools_x, y=ph_y, name="평균 pH"), row=2, col=1)

    fig_env.add_trace(go.Bar(x=schools_x, y=ec_target_y, name="목표 EC", offsetgroup=0), row=2, col=2)
    fig_env.add_trace(go.Bar(x=schools_x, y=ec_measured_y, name="실측 EC(평균)", offsetgroup=1), row=2, col=2)

    fig_env.update_layout(
        height=700,
        barmode="group",
        font=dict(family=PLOTLY_FONT_FAMILY),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0),
        margin=dict(l=30, r=30, t=60, b=80),
        template="plotly_white",
    )

    st.plotly_chart(fig_env, use_container_width=True)

    st.markdown("---")
    st.subheader("선택한 학교 시계열")

    # 전체 선택일 때는 탭 내부에서 시계열 학교를 고를 수 있게 제공
    if selected_school == "전체":
        ts_school = st.selectbox(
            "시계열로 볼 학교 선택",
            options=schools_all,
            index=0 if len(schools_all) == 0 else 0,
            key="ts_school_select",
        )
    else:
        ts_school = selected_school

    env_ts = env_df[env_df["학교"].apply(lambda x: _norm_eq(str(x), ts_school))].copy()
    if env_ts.empty:
        st.error("선택한 학교({})의 환경 데이터가 없습니다.".format(ts_school))
    else:
        env_ts = env_ts.sort_values("time")

        fig_t = px.line(
            env_ts,
            x="time",
            y="temperature",
            title="온도 변화",
            labels={"time": "시간", "temperature": "온도(°C)"},
        )
        fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), template="plotly_white")
        st.plotly_chart(fig_t, use_container_width=True)

        fig_h = px.line(
            env_ts,
            x="time",
            y="humidity",
            title="습도 변화",
            labels={"time": "시간", "humidity": "습도(%)"},
        )
        fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), template="plotly_white")
        st.plotly_chart(fig_h, use_container_width=True)

        fig_ec = px.line(
            env_ts,
            x="time",
            y="ec",
            title="EC 변화 (목표 EC 수평선 포함)",
            labels={"time": "시간", "ec": "EC"},
        )

        target_ec = _lookup_by_norm_key(EC_TARGET_BY_SCHOOL, ts_school)
        if target_ec is not None and pd.notna(target_ec):
            fig_ec.add_hline(
                y=float(target_ec),
                line_dash="dash",
                annotation_text="목표 EC {}".format(target_ec),
                annotation_position="top left",
            )

        fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), template="plotly_white")
        st.plotly_chart(fig_ec, use_container_width=True)

        with st.expander("환경 데이터 원본 테이블 / CSV 다운로드"):
            st.dataframe(env_ts, use_container_width=True, hide_index=True)

            csv_bytes = dataframe_to_csv_bytes(env_ts)
            file_name = "환경데이터_{}.csv".format(ts_school)
            st.download_button(
                label="CSV 다운로드",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
            )


# -----------------------------
# Tab 3: Growth results
# -----------------------------
with tab3:
    st.subheader("🥇 핵심 결과")

    ec_summary = growth_means_by_ec(growth_df)
    if ec_summary.empty:
        st.error("EC별 생육 요약을 계산할 수 없습니다. 생육 데이터/컬럼을 확인해주세요.")
        st.stop()

    best_ec_val = best_ec_from_growth(growth_df)
    best_weight = None
    if best_ec_val is not None:
        row = ec_summary[ec_summary["EC 목표"] == best_ec_val]
        if not row.empty:
            best_weight = float(row["mean_weight"].iloc[0]) if pd.notna(row["mean_weight"].iloc[0]) else None

    colA, colB = st.columns([1, 2])
    with colA:
        if best_ec_val is None:
            st.metric("최적 EC", "-")
        else:
            label = "최적 EC"
            if abs(best_ec_val - 2.0) < 1e-9:
                label = "최적 EC (하늘고, EC 2.0)"
            st.metric(label, "{:.1f}".format(best_ec_val), delta="평균 생중량 {:.3f} g".format(best_weight) if best_weight is not None else None)

    with colB:
        show_df = ec_summary.copy()
        show_df["EC 목표"] = show_df["EC 목표"].map(lambda x: "-" if pd.isna(x) else "{:.1f}".format(float(x)))
        show_df["평균 생중량(g)"] = show_df["mean_weight"].map(lambda x: "-" if pd.isna(x) else "{:.3f}".format(float(x)))
        show_df["평균 잎 수"] = show_df["mean_leaves"].map(lambda x: "-" if pd.isna(x) else "{:.2f}".format(float(x)))
        show_df["평균 지상부 길이(mm)"] = show_df["mean_shoot"].map(lambda x: "-" if pd.isna(x) else "{:.2f}".format(float(x)))
        show_df["개체수"] = show_df["count"].astype(int)
        show_df = show_df[["EC 목표", "평균 생중량(g)", "평균 잎 수", "평균 지상부 길이(mm)", "개체수"]]
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("EC별 생육 비교 (2x2)")

    ec_x = ec_summary["EC 목표"].tolist()
    w_y = ec_summary["mean_weight"].tolist()
    l_y = ec_summary["mean_leaves"].tolist()
    s_y = ec_summary["mean_shoot"].tolist()
    c_y = ec_summary["count"].tolist()

    fig_g = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수", "평균 지상부 길이(mm)", "개체수 비교"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig_g.add_trace(go.Bar(x=ec_x, y=w_y, name="평균 생중량"), row=1, col=1)
    fig_g.add_trace(go.Bar(x=ec_x, y=l_y, name="평균 잎 수"), row=1, col=2)
    fig_g.add_trace(go.Bar(x=ec_x, y=s_y, name="평균 지상부 길이"), row=2, col=1)
    fig_g.add_trace(go.Bar(x=ec_x, y=c_y, name="개체수"), row=2, col=2)

    # 최댓값(평균 생중량) 표시
    if ec_summary["mean_weight"].dropna().size > 0:
        idx = ec_summary["mean_weight"].idxmax()
        x_best = ec_summary.loc[idx, "EC 목표"]
        y_best = ec_summary.loc[idx, "mean_weight"]
        fig_g.add_trace(
            go.Scatter(
                x=[x_best],
                y=[y_best],
                mode="markers+text",
                text=["최댓값"],
                textposition="top center",
                name="최댓값(생중량)",
            ),
            row=1,
            col=1,
        )

    fig_g.update_layout(
        height=720,
        barmode="group",
        font=dict(family=PLOTLY_FONT_FAMILY),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0),
        margin=dict(l=30, r=30, t=60, b=80),
        template="plotly_white",
    )
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")
    st.subheader("학교별 생중량 분포")

    # 분포는 학교 비교 목적이므로 전체 기준 표시
    fig_dist = px.box(
        growth_df,
        x="학교",
        y="생중량(g)",
        points="outliers",
        title="학교별 생중량 분포 (Box Plot)",
        labels={"학교": "학교", "생중량(g)": "생중량(g)"},
    )
    fig_dist.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), template="plotly_white")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("상관관계 분석")

    fig_sc1 = px.scatter(
        growth_df if selected_school == "전체" else growth_scope,
        x="잎 수(장)",
        y="생중량(g)",
        color="학교" if selected_school == "전체" else None,
        title="잎 수 vs 생중량",
        labels={"잎 수(장)": "잎 수(장)", "생중량(g)": "생중량(g)"},
    )
    fig_sc1.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), template="plotly_white")
    st.plotly_chart(fig_sc1, use_container_width=True)

    fig_sc2 = px.scatter(
        growth_df if selected_school == "전체" else growth_scope,
        x="지상부 길이(mm)",
        y="생중량(g)",
        color="학교" if selected_school == "전체" else None,
        title="지상부 길이 vs 생중량",
        labels={"지상부 길이(mm)": "지상부 길이(mm)", "생중량(g)": "생중량(g)"},
    )
    fig_sc2.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), template="plotly_white")
    st.plotly_chart(fig_sc2, use_container_width=True)

    with st.expander("학교별 생육 데이터 원본 / XLSX 다운로드"):
        if selected_school == "전체":
            st.dataframe(growth_df, use_container_width=True, hide_index=True)
            download_df = growth_df
            file_tag = "전체"
        else:
            st.dataframe(growth_scope, use_container_width=True, hide_index=True)
            download_df = growth_scope
            file_tag = selected_school

        buffer = dataframe_to_xlsx_bytes_by_school(download_df)
        st.download_button(
            label="XLSX 다운로드",
            data=buffer,
            file_name="생육결과_{}.xlsx".format(file_tag),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
