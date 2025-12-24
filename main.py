import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="극지식물 생육 최적 EC 탐구 대시보드",
    layout="wide",
)

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

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_OPTIONS = ["전체"] + SCHOOLS

EC_CONDITION = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(강조)
    "아라고": 4.0,
    "동산고": 8.0,
}


# -----------------------------
# 유틸: 한글 NFC/NFD 안전 비교
# -----------------------------
def _norm_set(text: str) -> set:
    if text is None:
        return set()
    return {
        unicodedata.normalize("NFC", str(text)),
        unicodedata.normalize("NFD", str(text)),
    }


def _token_in_text(text: str, token: str) -> bool:
    text_variants = _norm_set(text)
    token_variants = _norm_set(token)
    for tv in text_variants:
        for kv in token_variants:
            if kv in tv:
                return True
    return False


def _best_school_match(name_or_sheet: str) -> str | None:
    # 파일명/시트명에서 학교명을 "포함 여부"로 찾기 (NFC/NFD 양방향)
    for sch in SCHOOLS:
        if _token_in_text(name_or_sheet, sch):
            return sch
    return None


def _safe_sheet_name(name: str) -> str:
    # 엑셀 시트명 제한(31자) + 기본 정리
    base = unicodedata.normalize("NFC", str(name)).strip()
    if not base:
        base = "data"
    return base[:31]


# -----------------------------
# 파일 탐색 (iterdir + NFC/NFD)
# -----------------------------
@st.cache_data(show_spinner=False)
def discover_assets(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    env_csv_by_school: dict[str, str] = {}
    xlsx_path: str = ""

    if not data_dir.exists():
        return {"env_csv_by_school": {}, "growth_xlsx": ""}

    csv_candidates = []
    xlsx_candidates = []

    for p in data_dir.iterdir():  # ✅ 필수: iterdir()
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            csv_candidates.append(p)
        elif suf in [".xlsx", ".xlsm", ".xls"]:
            xlsx_candidates.append(p)

    # 환경 CSV: 학교명 포함 + "환경" "데이터" 토큰 포함 (하드코딩된 전체 파일명 사용 X)
    for p in csv_candidates:
        nm = p.name
        if _token_in_text(nm, "환경") and _token_in_text(nm, "데이터"):
            sch = _best_school_match(nm)
            if sch is None:
                # 혹시 "학교명_" 형태가 아니더라도 대응 (스플릿은 보조적으로만 사용)
                stem = unicodedata.normalize("NFC", p.stem)
                parts = stem.split("_")
                if parts:
                    candidate = parts[0].strip()
                    # candidate가 실제 학교 옵션 중 하나면 채택
                    for s in SCHOOLS:
                        if _token_in_text(candidate, s) or _token_in_text(s, candidate):
                            sch = s
                            break
            if sch is not None:
                env_csv_by_school[sch] = str(p)

    # 생육결과 XLSX: "생육" + "결과" 또는 "생육" + "데이터" 같은 토큰 기반 탐지
    for p in xlsx_candidates:
        nm = p.name
        if (_token_in_text(nm, "생육") and _token_in_text(nm, "결과")) or (
            _token_in_text(nm, "생육") and _token_in_text(nm, "데이터")
        ):
            xlsx_path = str(p)
            break
    if (not xlsx_path) and xlsx_candidates:
        # 토큰 매칭이 실패했더라도 xlsx가 1개뿐인 상황을 대비한 안전 fallback
        xlsx_path = str(xlsx_candidates[0])

    return {"env_csv_by_school": env_csv_by_school, "growth_xlsx": xlsx_path}


# -----------------------------
# 데이터 로딩 (cache + 안전 인코딩)
# -----------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    # 학교별 CSV 인코딩이 다를 수 있어 utf-8-sig -> cp949 순으로 시도
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막: pandas 기본
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_env_csv(path: str, school: str) -> pd.DataFrame:
    df = _read_csv_safely(path)

    # 컬럼 표준화
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["time", "temperature", "humidity", "ph", "ec"]
    for c in needed:
        if c not in df.columns:
            # 일부 파일에서 대문자/다른 표기일 수 있어 유사 검색
            for original in df.columns:
                if _token_in_text(original, c):
                    df = df.rename(columns={original: c})
                    break

    missing = [c for c in needed if c not in df.columns]
    if missing:
        # 필수 컬럼이 없으면 빈 DF 반환 (상위에서 st.error)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df["학교"] = school
    df["EC조건"] = EC_CONDITION.get(school, np.nan)
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    frames = []
    for sheet in xl.sheet_names:  # ✅ 시트명 하드코딩 금지: 실제 sheet_names 사용
        sch = _best_school_match(sheet)
        if sch is None:
            continue

        try:
            df = xl.parse(sheet_name=sheet)
        except Exception:
            continue

        # 컬럼명 트림
        df.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df.columns]

        # 기대 컬럼(한글) 기반으로 숫자형 변환
        numeric_like = []
        for c in df.columns:
            if c == "개체번호":
                continue
            numeric_like.append(c)

        for c in numeric_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["학교"] = sch
        df["EC조건"] = EC_CONDITION.get(sch, np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# -----------------------------
# 다운로드 유틸 (BytesIO + openpyxl)
# -----------------------------
def to_excel_bytes_single(df: pd.DataFrame, sheet_name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    safe = _safe_sheet_name(sheet_name)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


def to_excel_bytes_multi(sheets: dict[str, pd.DataFrame]) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = _safe_sheet_name(name)
            df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


# -----------------------------
# 분석 유틸
# -----------------------------
def env_summary(df_env_all: pd.DataFrame) -> pd.DataFrame:
    if df_env_all.empty:
        return pd.DataFrame()

    g = df_env_all.groupby("학교", dropna=False)
    out = g.agg(
        측정수=("time", "count"),
        평균온도=("temperature", "mean"),
        평균습도=("humidity", "mean"),
        평균pH=("ph", "mean"),
        평균EC=("ec", "mean"),
    ).reset_index()

    for c in out.columns:
        if c != "학교" and c != "측정수":
            out[c] = out[c].round(3)
    return out


def growth_summary(df_growth_all: pd.DataFrame) -> pd.DataFrame:
    if df_growth_all.empty:
        return pd.DataFrame()

    # 존재하는 지표만 요약
    candidates = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    metrics = [c for c in candidates if c in df_growth_all.columns]

    agg_dict = {}
    for m in metrics:
        agg_dict[m + "_평균"] = (m, "mean")
        agg_dict[m + "_표준편차"] = (m, "std")

    out = df_growth_all.groupby(["학교", "EC조건"], dropna=False).agg(**agg_dict).reset_index()

    # 반올림
    for c in out.columns:
        if c not in ["학교", "EC조건"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out


def plotly_apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    return fig


# -----------------------------
# UI 시작
# -----------------------------
st.title("극지식물 생육 최적 EC 탐구 대시보드")

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

with st.spinner("데이터 파일 탐색 중..."):
    assets = discover_assets(str(data_dir))

env_paths = assets.get("env_csv_by_school", {}) or {}
growth_xlsx = assets.get("growth_xlsx", "") or ""

# 사이드바
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", options=SCHOOL_OPTIONS, index=0)

# 데이터 로딩
df_env_all_list = []
if env_paths:
    with st.spinner("환경 데이터(CSV) 로딩 중..."):
        for sch, pth in env_paths.items():
            try:
                dfi = load_env_csv(pth, sch)
                if not dfi.empty:
                    df_env_all_list.append(dfi)
            except Exception:
                # 개별 파일 오류는 전체 중단 대신 경고로 처리
                st.sidebar.warning("환경 데이터 일부를 읽지 못했습니다: " + unicodedata.normalize("NFC", sch))

df_env_all = pd.concat(df_env_all_list, ignore_index=True) if df_env_all_list else pd.DataFrame()

with st.spinner("생육 결과 데이터(XLSX) 로딩 중..."):
    df_growth_all = load_growth_xlsx(growth_xlsx)

# 유효성 안내
if not env_paths:
    st.error("data 폴더에서 환경 CSV 파일을 찾지 못했습니다. (예: '...환경...데이터...csv')")
if not growth_xlsx:
    st.error("data 폴더에서 생육 결과 XLSX 파일을 찾지 못했습니다. (예: '...생육...결과...xlsx')")

if df_env_all.empty:
    st.error("환경 데이터가 비어있거나 필수 컬럼(time, temperature, humidity, ph, ec)을 찾지 못했습니다.")
if df_growth_all.empty:
    st.error("생육 결과 데이터가 비어있거나 시트에서 학교명을 인식하지 못했습니다.")

# 선택 필터
if selected_school != "전체":
    df_env = df_env_all[df_env_all["학교"] == selected_school].copy() if not df_env_all.empty else pd.DataFrame()
    df_growth = df_growth_all[df_growth_all["학교"] == selected_school].copy() if not df_growth_all.empty else pd.DataFrame()
else:
    df_env = df_env_all.copy()
    df_growth = df_growth_all.copy()

tabs = st.tabs(
    [
        "탭1: 극지식물 최적 EC 농도 연구 대시보드",
        "탭2: EC 농도에 따른 극지식물 생육 비교 분석",
        "탭3: 4개교 데이터로 찾는 최적 EC (1.0~8.0)",
    ]
)

# -----------------------------
# 탭1
# -----------------------------
with tabs[0]:
    st.subheader("학교별 생육환경 비교 (환경 데이터)")

    if df_env.empty:
        st.error("선택된 조건에서 표시할 환경 데이터가 없습니다.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("환경 측정 레코드 수", f"{len(df_env):,}")
        colB.metric("평균 온도", f"{df_env['temperature'].mean():.2f}")
        colC.metric("평균 습도", f"{df_env['humidity'].mean():.2f}")
        colD.metric("평균 pH", f"{df_env['ph'].mean():.2f}")

        st.caption("※ 학교별 측정 주기가 달라 시간축 비교 시 주의하세요.")

        # (1) 시계열 서브플롯
        fig_ts = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("온도(temperature)", "습도(humidity)", "pH(ph)", "EC(ec)"),
        )

        for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
            dfi = df_env[df_env["학교"] == sch].copy()
            if dfi.empty:
                continue

            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["temperature"], mode="lines", name=sch), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["humidity"], mode="lines", name=sch, showlegend=False), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ph"], mode="lines", name=sch, showlegend=False), row=3, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ec"], mode="lines", name=sch, showlegend=False), row=4, col=1)

        fig_ts.update_height(900)
        fig_ts.update_layout(
            margin=dict(l=30, r=30, t=70, b=30),
            legend_title_text="학교",
        )
        fig_ts = plotly_apply_font(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

        # (2) 분포 비교(박스플롯) - 학교별 비교가 핵심이므로 전체일 때 가치가 큼
        if selected_school == "전체":
            st.subheader("학교별 환경 분포 비교")
            fig_box = make_subplots(rows=2, cols=2, subplot_titles=("온도", "습도", "pH", "EC"))
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["temperature"], name="온도"), row=1, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["humidity"], name="습도", showlegend=False), row=1, col=2)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ph"], name="pH", showlegend=False), row=2, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ec"], name="EC", showlegend=False), row=2, col=2)
            fig_box.update_height(700)
            fig_box.update_layout(margin=dict(l=30, r=30, t=70, b=30))
            fig_box = plotly_apply_font(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        # (3) 환경 요약표 + 다운로드
        st.subheader("환경 요약(학교별 평균)")
        df_env_sum = env_summary(df_env_all)
        if df_env_sum.empty:
            st.error("환경 요약을 만들 수 없습니다.")
        else:
            st.dataframe(df_env_sum, use_container_width=True)

            buf = to_excel_bytes_single(df_env_sum, "환경요약")
            st.download_button(
                label="환경 요약 XLSX 다운로드",
                data=buf,
                file_name="환경요약.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("원본 환경 데이터 다운로드")
        buf2 = to_excel_bytes_single(df_env, "환경데이터")
        # 파일명 f-string 금지 → 문자열 덧셈 사용
        down_name = "전체_환경데이터.xlsx" if selected_school == "전체" else (selected_school + "_환경데이터.xlsx")
        st.download_button(
            label="선택된 환경 데이터 XLSX 다운로드",
            data=buf2,
            file_name=down_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# -----------------------------
# 탭2
# -----------------------------
with tabs[1]:
    st.subheader("EC 농도에 따른 극지식물 생육 비교 (생육 결과)")

    if df_growth.empty:
        st.error("선택된 조건에서 표시할 생육 결과 데이터가 없습니다.")
    else:
        # 생육 지표 선택
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth.columns]

        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("비교할 생육 지표 선택", options=metric_candidates, index=0)

            # 학교(=EC조건)별 분포
            st.caption("※ 각 학교는 서로 다른 EC 조건(1.0, 2.0, 4.0, 8.0)에서 재배한 결과입니다.")
            df_plot = df_growth.dropna(subset=[metric, "EC조건"]).copy()

            if df_plot.empty:
                st.error("선택한 지표에 대해 유효한 값이 없습니다.")
            else:
                # 박스플롯: x=EC, color/hover=학교
                fig = go.Figure()

                # 하늘고(EC 2.0) 강조: 두꺼운 라인 + 주석
                for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
                    dfi = df_plot[df_plot["학교"] == sch]
                    if dfi.empty:
                        continue

                    is_best = (sch == "하늘고")
                    fig.add_trace(
                        go.Box(
                            x=dfi["EC조건"],
                            y=dfi[metric],
                            name=sch,
                            boxmean="sd",
                            line=dict(width=4 if is_best else 2),
                        )
                    )

                fig.update_layout(
                    xaxis_title="EC 조건",
                    yaxis_title=metric,
                    margin=dict(l=30, r=30, t=60, b=30),
                )
                fig = plotly_apply_font(fig)

                # 하늘고(EC 2.0) 최적 강조: 세로 하이라이트 영역
                fig.add_vrect(
                    x0=1.8, x1=2.2,
                    fillcolor="rgba(0, 0, 0, 0.08)",
                    line_width=0,
                    annotation_text="하늘고(EC 2.0) 최적(강조)",
                    annotation_position="top left",
                )

                st.plotly_chart(fig, use_container_width=True)

                # 요약표
                st.subheader("학교(EC)별 요약 통계")
                df_sum = growth_summary(df_growth_all if selected_school == "전체" else df_growth)
                if df_sum.empty:
                    st.error("요약 통계를 만들 수 없습니다.")
                else:
                    st.dataframe(df_sum, use_container_width=True)

                    buf = to_excel_bytes_single(df_sum, "생육요약")
                    down_name = "전체_생육요약.xlsx" if selected_school == "전체" else (selected_school + "_생육요약.xlsx")
                    st.download_button(
                        label="생육 요약 XLSX 다운로드",
                        data=buf,
                        file_name=down_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # 환경요소 vs 생육(학교 평균 기반 상관) — 표본이 4개(학교 4개)라 참고용임을 명시
                st.subheader("환경 요소 영향(참고): 학교 평균 기반 상관 비교")
                if df_env_all.empty:
                    st.error("환경 데이터가 없어 상관 분석을 진행할 수 없습니다.")
                else:
                    # 학교별 평균 환경 + 학교별 평균 생육(선택 metric)
                    env_avg = df_env_all.groupby("학교")[["temperature", "humidity", "ph", "ec"]].mean().reset_index()
                    grow_avg = df_growth_all.groupby("학교")[[metric]].mean().reset_index()

                    merged = pd.merge(env_avg, grow_avg, on="학교", how="inner")
                    merged = merged.dropna()

                    if len(merged) < 3:
                        st.error("상관을 계산하기에 데이터가 부족합니다.")
                    else:
                        corr_cols = ["temperature", "humidity", "ph", "ec", metric]
                        corr = merged[corr_cols].corr()

                        fig_hm = go.Figure(
                            data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.index,
                                zmin=-1,
                                zmax=1,
                                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
                            )
                        )
                        fig_hm.update_layout(
                            height=520,
                            margin=dict(l=30, r=30, t=30, b=30),
                        )
                        fig_hm = plotly_apply_font(fig_hm)
                        st.plotly_chart(fig_hm, use_container_width=True)

                        # 어떤 요소가 더 "세게" 움직였는지: |corr| 비교 (참고)
                        strength = corr[metric].drop(metric).abs().sort_values(ascending=False)
                        st.write("**(참고) 선택 지표와의 |상관계수| 순위**")
                        st.dataframe(strength.rename("abs(corr)"), use_container_width=True)

                st.subheader("원본 생육 데이터 다운로드")
                buf2 = to_excel_bytes_single(df_growth, "생육데이터")
                down_name2 = "전체_생육데이터.xlsx" if selected_school == "전체" else (selected_school + "_생육데이터.xlsx")
                st.download_button(
                    label="선택된 생육 데이터 XLSX 다운로드",
                    data=buf2,
                    file_name=down_name2,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# -----------------------------
# 탭3
# -----------------------------
with tabs[2]:
    st.subheader("4개교 데이터로 최적 EC 추정 (1.0 ~ 8.0)")

    if df_growth_all.empty:
        st.error("생육 결과 데이터가 없어 최적 EC 추정을 할 수 없습니다.")
    else:
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth_all.columns]
        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("최적화를 보고 싶은 지표", options=metric_candidates, index=0)
            st.caption("※ 학교=EC 조건이므로, 4개 점(EC 1/2/4/8) 기반으로 곡선을 ‘참고용’으로 적합합니다.")

            df_use = df_growth_all.dropna(subset=[metric, "EC조건"]).copy()
            if df_use.empty:
                st.error("선택한 지표에 대한 유효 데이터가 없습니다.")
            else:
                # EC별 평균/표준편차
                grp = df_use.groupby(["학교", "EC조건"])[metric].agg(["mean", "std", "count"]).reset_index()
                grp = grp.sort_values("EC조건")

                st.write("### EC별 평균 비교")
                st.dataframe(
                    grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"}),
                    use_container_width=True,
                )

                # 생중량이면 하늘고 최적 강조 문구
                if metric == "생중량(g)":
                    st.info("하늘고(EC 2.0)가 **최적** 조건으로 제시되어 있으므로 그래프에서 강조 표시합니다.")

                # 곡선 적합(2차) + 최적점(-b/2a)
                xs = grp["EC조건"].astype(float).values
                ys = grp["mean"].astype(float).values

                predicted_opt = None
                fit_ok = False

                if len(np.unique(xs)) >= 3 and len(xs) >= 3:
                    try:
                        coef = np.polyfit(xs, ys, deg=2)  # a, b, c
                        a, b, c = coef[0], coef[1], coef[2]
                        if a != 0:
                            x_opt = -b / (2 * a)
                            predicted_opt = float(x_opt)
                            fit_ok = True
                    except Exception:
                        fit_ok = False

                # 그래프
                fig = go.Figure()

                # 점(평균) + 에러바
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        name="EC별 평균",
                        error_y=dict(
                            type="data",
                            array=np.nan_to_num(grp["std"].values, nan=0.0),
                            visible=True,
                        ),
                        hovertemplate="EC=%{x}<br>평균=%{y:.4f}<extra></extra>",
                    )
                )

                # 하늘고(EC 2.0) 강조 마커(겹쳐서 표시)
                if 2.0 in xs:
                    y_2 = float(grp.loc[grp["EC조건"] == 2.0, "mean"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[2.0],
                            y=[y_2],
                            mode="markers",
                            name="하늘고(EC 2.0) 강조",
                            marker=dict(size=16, symbol="star"),
                            hovertemplate="하늘고(EC 2.0)<br>평균=%{y:.4f}<extra></extra>",
                        )
                    )

                # 적합 곡선
                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="2차 적합(참고)",
                            hoverinfo="skip",
                        )
                    )

                    # 예측 최적 EC 표시(범위 내로만 안내)
                    fig.add_vline(
                        x=predicted_opt,
                        line_dash="dash",
                        annotation_text="예측 최적 EC(참고)",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    xaxis_title="EC",
                    yaxis_title=metric + " (평균)",
                    margin=dict(l=30, r=30, t=60, b=30),
                    height=600,
                )
                fig = plotly_apply_font(fig)
                st.plotly_chart(fig, use_container_width=True)

                # 해석 요약
                st.write("### 해석 요약")
                if fit_ok and predicted_opt is not None:
                    st.write(
                        "- 2차 적합 기반 예측 최적 EC(참고): **"
                        + f"{predicted_opt:.2f}"
                        + "**"
                    )
                    st.write("- 단, 학교 4개 점으로 적합한 결과라 **참고용**이며, 반복 실험/중간 EC(예: 1.5, 2.5, 3.0 등) 검증이 필요합니다.")
                else:
                    st.write("- 현재 데이터로는 안정적인 곡선 적합이 어려워(값 부족/오류) 예측 최적 EC를 계산하지 않았습니다.")

                # 다운로드: 요약 + (선택) 예측 곡선 점
                st.subheader("분석 결과 다운로드")
                sheets = {}
                sheets["EC별요약"] = grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"})

                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    df_curve = pd.DataFrame({"EC": x_line, "예측값(2차적합)": y_line})
                    sheets["2차적합곡선"] = df_curve

                buffer = to_excel_bytes_multi(sheets)
                st.download_button(
                    label="최적 EC 분석 XLSX 다운로드",
                    data=buffer,
                    file_name="최적EC_분석결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="극지식물 생육 최적 EC 탐구 대시보드",
    layout="wide",
)

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

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_OPTIONS = ["전체"] + SCHOOLS

EC_CONDITION = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(강조)
    "아라고": 4.0,
    "동산고": 8.0,
}


# -----------------------------
# 유틸: 한글 NFC/NFD 안전 비교
# -----------------------------
def _norm_set(text: str) -> set:
    if text is None:
        return set()
    return {
        unicodedata.normalize("NFC", str(text)),
        unicodedata.normalize("NFD", str(text)),
    }


def _token_in_text(text: str, token: str) -> bool:
    text_variants = _norm_set(text)
    token_variants = _norm_set(token)
    for tv in text_variants:
        for kv in token_variants:
            if kv in tv:
                return True
    return False


def _best_school_match(name_or_sheet: str) -> str | None:
    # 파일명/시트명에서 학교명을 "포함 여부"로 찾기 (NFC/NFD 양방향)
    for sch in SCHOOLS:
        if _token_in_text(name_or_sheet, sch):
            return sch
    return None


def _safe_sheet_name(name: str) -> str:
    # 엑셀 시트명 제한(31자) + 기본 정리
    base = unicodedata.normalize("NFC", str(name)).strip()
    if not base:
        base = "data"
    return base[:31]


# -----------------------------
# 파일 탐색 (iterdir + NFC/NFD)
# -----------------------------
@st.cache_data(show_spinner=False)
def discover_assets(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    env_csv_by_school: dict[str, str] = {}
    xlsx_path: str = ""

    if not data_dir.exists():
        return {"env_csv_by_school": {}, "growth_xlsx": ""}

    csv_candidates = []
    xlsx_candidates = []

    for p in data_dir.iterdir():  # ✅ 필수: iterdir()
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            csv_candidates.append(p)
        elif suf in [".xlsx", ".xlsm", ".xls"]:
            xlsx_candidates.append(p)

    # 환경 CSV: 학교명 포함 + "환경" "데이터" 토큰 포함 (하드코딩된 전체 파일명 사용 X)
    for p in csv_candidates:
        nm = p.name
        if _token_in_text(nm, "환경") and _token_in_text(nm, "데이터"):
            sch = _best_school_match(nm)
            if sch is None:
                # 혹시 "학교명_" 형태가 아니더라도 대응 (스플릿은 보조적으로만 사용)
                stem = unicodedata.normalize("NFC", p.stem)
                parts = stem.split("_")
                if parts:
                    candidate = parts[0].strip()
                    # candidate가 실제 학교 옵션 중 하나면 채택
                    for s in SCHOOLS:
                        if _token_in_text(candidate, s) or _token_in_text(s, candidate):
                            sch = s
                            break
            if sch is not None:
                env_csv_by_school[sch] = str(p)

    # 생육결과 XLSX: "생육" + "결과" 또는 "생육" + "데이터" 같은 토큰 기반 탐지
    for p in xlsx_candidates:
        nm = p.name
        if (_token_in_text(nm, "생육") and _token_in_text(nm, "결과")) or (
            _token_in_text(nm, "생육") and _token_in_text(nm, "데이터")
        ):
            xlsx_path = str(p)
            break
    if (not xlsx_path) and xlsx_candidates:
        # 토큰 매칭이 실패했더라도 xlsx가 1개뿐인 상황을 대비한 안전 fallback
        xlsx_path = str(xlsx_candidates[0])

    return {"env_csv_by_school": env_csv_by_school, "growth_xlsx": xlsx_path}


# -----------------------------
# 데이터 로딩 (cache + 안전 인코딩)
# -----------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    # 학교별 CSV 인코딩이 다를 수 있어 utf-8-sig -> cp949 순으로 시도
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막: pandas 기본
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_env_csv(path: str, school: str) -> pd.DataFrame:
    df = _read_csv_safely(path)

    # 컬럼 표준화
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["time", "temperature", "humidity", "ph", "ec"]
    for c in needed:
        if c not in df.columns:
            # 일부 파일에서 대문자/다른 표기일 수 있어 유사 검색
            for original in df.columns:
                if _token_in_text(original, c):
                    df = df.rename(columns={original: c})
                    break

    missing = [c for c in needed if c not in df.columns]
    if missing:
        # 필수 컬럼이 없으면 빈 DF 반환 (상위에서 st.error)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df["학교"] = school
    df["EC조건"] = EC_CONDITION.get(school, np.nan)
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    frames = []
    for sheet in xl.sheet_names:  # ✅ 시트명 하드코딩 금지: 실제 sheet_names 사용
        sch = _best_school_match(sheet)
        if sch is None:
            continue

        try:
            df = xl.parse(sheet_name=sheet)
        except Exception:
            continue

        # 컬럼명 트림
        df.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df.columns]

        # 기대 컬럼(한글) 기반으로 숫자형 변환
        numeric_like = []
        for c in df.columns:
            if c == "개체번호":
                continue
            numeric_like.append(c)

        for c in numeric_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["학교"] = sch
        df["EC조건"] = EC_CONDITION.get(sch, np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# -----------------------------
# 다운로드 유틸 (BytesIO + openpyxl)
# -----------------------------
def to_excel_bytes_single(df: pd.DataFrame, sheet_name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    safe = _safe_sheet_name(sheet_name)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


def to_excel_bytes_multi(sheets: dict[str, pd.DataFrame]) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = _safe_sheet_name(name)
            df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


# -----------------------------
# 분석 유틸
# -----------------------------
def env_summary(df_env_all: pd.DataFrame) -> pd.DataFrame:
    if df_env_all.empty:
        return pd.DataFrame()

    g = df_env_all.groupby("학교", dropna=False)
    out = g.agg(
        측정수=("time", "count"),
        평균온도=("temperature", "mean"),
        평균습도=("humidity", "mean"),
        평균pH=("ph", "mean"),
        평균EC=("ec", "mean"),
    ).reset_index()

    for c in out.columns:
        if c != "학교" and c != "측정수":
            out[c] = out[c].round(3)
    return out


def growth_summary(df_growth_all: pd.DataFrame) -> pd.DataFrame:
    if df_growth_all.empty:
        return pd.DataFrame()

    # 존재하는 지표만 요약
    candidates = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    metrics = [c for c in candidates if c in df_growth_all.columns]

    agg_dict = {}
    for m in metrics:
        agg_dict[m + "_평균"] = (m, "mean")
        agg_dict[m + "_표준편차"] = (m, "std")

    out = df_growth_all.groupby(["학교", "EC조건"], dropna=False).agg(**agg_dict).reset_index()

    # 반올림
    for c in out.columns:
        if c not in ["학교", "EC조건"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out


def plotly_apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    return fig


# -----------------------------
# UI 시작
# -----------------------------
st.title("극지식물 생육 최적 EC 탐구 대시보드")

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

with st.spinner("데이터 파일 탐색 중..."):
    assets = discover_assets(str(data_dir))

env_paths = assets.get("env_csv_by_school", {}) or {}
growth_xlsx = assets.get("growth_xlsx", "") or ""

# 사이드바
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", options=SCHOOL_OPTIONS, index=0)

# 데이터 로딩
df_env_all_list = []
if env_paths:
    with st.spinner("환경 데이터(CSV) 로딩 중..."):
        for sch, pth in env_paths.items():
            try:
                dfi = load_env_csv(pth, sch)
                if not dfi.empty:
                    df_env_all_list.append(dfi)
            except Exception:
                # 개별 파일 오류는 전체 중단 대신 경고로 처리
                st.sidebar.warning("환경 데이터 일부를 읽지 못했습니다: " + unicodedata.normalize("NFC", sch))

df_env_all = pd.concat(df_env_all_list, ignore_index=True) if df_env_all_list else pd.DataFrame()

with st.spinner("생육 결과 데이터(XLSX) 로딩 중..."):
    df_growth_all = load_growth_xlsx(growth_xlsx)

# 유효성 안내
if not env_paths:
    st.error("data 폴더에서 환경 CSV 파일을 찾지 못했습니다. (예: '...환경...데이터...csv')")
if not growth_xlsx:
    st.error("data 폴더에서 생육 결과 XLSX 파일을 찾지 못했습니다. (예: '...생육...결과...xlsx')")

if df_env_all.empty:
    st.error("환경 데이터가 비어있거나 필수 컬럼(time, temperature, humidity, ph, ec)을 찾지 못했습니다.")
if df_growth_all.empty:
    st.error("생육 결과 데이터가 비어있거나 시트에서 학교명을 인식하지 못했습니다.")

# 선택 필터
if selected_school != "전체":
    df_env = df_env_all[df_env_all["학교"] == selected_school].copy() if not df_env_all.empty else pd.DataFrame()
    df_growth = df_growth_all[df_growth_all["학교"] == selected_school].copy() if not df_growth_all.empty else pd.DataFrame()
else:
    df_env = df_env_all.copy()
    df_growth = df_growth_all.copy()

tabs = st.tabs(
    [
        "탭1: 극지식물 최적 EC 농도 연구 대시보드",
        "탭2: EC 농도에 따른 극지식물 생육 비교 분석",
        "탭3: 4개교 데이터로 찾는 최적 EC (1.0~8.0)",
    ]
)

# -----------------------------
# 탭1
# -----------------------------
with tabs[0]:
    st.subheader("학교별 생육환경 비교 (환경 데이터)")

    if df_env.empty:
        st.error("선택된 조건에서 표시할 환경 데이터가 없습니다.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("환경 측정 레코드 수", f"{len(df_env):,}")
        colB.metric("평균 온도", f"{df_env['temperature'].mean():.2f}")
        colC.metric("평균 습도", f"{df_env['humidity'].mean():.2f}")
        colD.metric("평균 pH", f"{df_env['ph'].mean():.2f}")

        st.caption("※ 학교별 측정 주기가 달라 시간축 비교 시 주의하세요.")

        # (1) 시계열 서브플롯
        fig_ts = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("온도(temperature)", "습도(humidity)", "pH(ph)", "EC(ec)"),
        )

        for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
            dfi = df_env[df_env["학교"] == sch].copy()
            if dfi.empty:
                continue

            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["temperature"], mode="lines", name=sch), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["humidity"], mode="lines", name=sch, showlegend=False), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ph"], mode="lines", name=sch, showlegend=False), row=3, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ec"], mode="lines", name=sch, showlegend=False), row=4, col=1)

        fig_ts.update_height(900)
        fig_ts.update_layout(
            margin=dict(l=30, r=30, t=70, b=30),
            legend_title_text="학교",
        )
        fig_ts = plotly_apply_font(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

        # (2) 분포 비교(박스플롯) - 학교별 비교가 핵심이므로 전체일 때 가치가 큼
        if selected_school == "전체":
            st.subheader("학교별 환경 분포 비교")
            fig_box = make_subplots(rows=2, cols=2, subplot_titles=("온도", "습도", "pH", "EC"))
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["temperature"], name="온도"), row=1, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["humidity"], name="습도", showlegend=False), row=1, col=2)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ph"], name="pH", showlegend=False), row=2, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ec"], name="EC", showlegend=False), row=2, col=2)
            fig_box.update_height(700)
            fig_box.update_layout(margin=dict(l=30, r=30, t=70, b=30))
            fig_box = plotly_apply_font(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        # (3) 환경 요약표 + 다운로드
        st.subheader("환경 요약(학교별 평균)")
        df_env_sum = env_summary(df_env_all)
        if df_env_sum.empty:
            st.error("환경 요약을 만들 수 없습니다.")
        else:
            st.dataframe(df_env_sum, use_container_width=True)

            buf = to_excel_bytes_single(df_env_sum, "환경요약")
            st.download_button(
                label="환경 요약 XLSX 다운로드",
                data=buf,
                file_name="환경요약.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("원본 환경 데이터 다운로드")
        buf2 = to_excel_bytes_single(df_env, "환경데이터")
        # 파일명 f-string 금지 → 문자열 덧셈 사용
        down_name = "전체_환경데이터.xlsx" if selected_school == "전체" else (selected_school + "_환경데이터.xlsx")
        st.download_button(
            label="선택된 환경 데이터 XLSX 다운로드",
            data=buf2,
            file_name=down_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# -----------------------------
# 탭2
# -----------------------------
with tabs[1]:
    st.subheader("EC 농도에 따른 극지식물 생육 비교 (생육 결과)")

    if df_growth.empty:
        st.error("선택된 조건에서 표시할 생육 결과 데이터가 없습니다.")
    else:
        # 생육 지표 선택
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth.columns]

        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("비교할 생육 지표 선택", options=metric_candidates, index=0)

            # 학교(=EC조건)별 분포
            st.caption("※ 각 학교는 서로 다른 EC 조건(1.0, 2.0, 4.0, 8.0)에서 재배한 결과입니다.")
            df_plot = df_growth.dropna(subset=[metric, "EC조건"]).copy()

            if df_plot.empty:
                st.error("선택한 지표에 대해 유효한 값이 없습니다.")
            else:
                # 박스플롯: x=EC, color/hover=학교
                fig = go.Figure()

                # 하늘고(EC 2.0) 강조: 두꺼운 라인 + 주석
                for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
                    dfi = df_plot[df_plot["학교"] == sch]
                    if dfi.empty:
                        continue

                    is_best = (sch == "하늘고")
                    fig.add_trace(
                        go.Box(
                            x=dfi["EC조건"],
                            y=dfi[metric],
                            name=sch,
                            boxmean="sd",
                            line=dict(width=4 if is_best else 2),
                        )
                    )

                fig.update_layout(
                    xaxis_title="EC 조건",
                    yaxis_title=metric,
                    margin=dict(l=30, r=30, t=60, b=30),
                )
                fig = plotly_apply_font(fig)

                # 하늘고(EC 2.0) 최적 강조: 세로 하이라이트 영역
                fig.add_vrect(
                    x0=1.8, x1=2.2,
                    fillcolor="rgba(0, 0, 0, 0.08)",
                    line_width=0,
                    annotation_text="하늘고(EC 2.0) 최적(강조)",
                    annotation_position="top left",
                )

                st.plotly_chart(fig, use_container_width=True)

                # 요약표
                st.subheader("학교(EC)별 요약 통계")
                df_sum = growth_summary(df_growth_all if selected_school == "전체" else df_growth)
                if df_sum.empty:
                    st.error("요약 통계를 만들 수 없습니다.")
                else:
                    st.dataframe(df_sum, use_container_width=True)

                    buf = to_excel_bytes_single(df_sum, "생육요약")
                    down_name = "전체_생육요약.xlsx" if selected_school == "전체" else (selected_school + "_생육요약.xlsx")
                    st.download_button(
                        label="생육 요약 XLSX 다운로드",
                        data=buf,
                        file_name=down_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # 환경요소 vs 생육(학교 평균 기반 상관) — 표본이 4개(학교 4개)라 참고용임을 명시
                st.subheader("환경 요소 영향(참고): 학교 평균 기반 상관 비교")
                if df_env_all.empty:
                    st.error("환경 데이터가 없어 상관 분석을 진행할 수 없습니다.")
                else:
                    # 학교별 평균 환경 + 학교별 평균 생육(선택 metric)
                    env_avg = df_env_all.groupby("학교")[["temperature", "humidity", "ph", "ec"]].mean().reset_index()
                    grow_avg = df_growth_all.groupby("학교")[[metric]].mean().reset_index()

                    merged = pd.merge(env_avg, grow_avg, on="학교", how="inner")
                    merged = merged.dropna()

                    if len(merged) < 3:
                        st.error("상관을 계산하기에 데이터가 부족합니다.")
                    else:
                        corr_cols = ["temperature", "humidity", "ph", "ec", metric]
                        corr = merged[corr_cols].corr()

                        fig_hm = go.Figure(
                            data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.index,
                                zmin=-1,
                                zmax=1,
                                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
                            )
                        )
                        fig_hm.update_layout(
                            height=520,
                            margin=dict(l=30, r=30, t=30, b=30),
                        )
                        fig_hm = plotly_apply_font(fig_hm)
                        st.plotly_chart(fig_hm, use_container_width=True)

                        # 어떤 요소가 더 "세게" 움직였는지: |corr| 비교 (참고)
                        strength = corr[metric].drop(metric).abs().sort_values(ascending=False)
                        st.write("**(참고) 선택 지표와의 |상관계수| 순위**")
                        st.dataframe(strength.rename("abs(corr)"), use_container_width=True)

                st.subheader("원본 생육 데이터 다운로드")
                buf2 = to_excel_bytes_single(df_growth, "생육데이터")
                down_name2 = "전체_생육데이터.xlsx" if selected_school == "전체" else (selected_school + "_생육데이터.xlsx")
                st.download_button(
                    label="선택된 생육 데이터 XLSX 다운로드",
                    data=buf2,
                    file_name=down_name2,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# -----------------------------
# 탭3
# -----------------------------
with tabs[2]:
    st.subheader("4개교 데이터로 최적 EC 추정 (1.0 ~ 8.0)")

    if df_growth_all.empty:
        st.error("생육 결과 데이터가 없어 최적 EC 추정을 할 수 없습니다.")
    else:
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth_all.columns]
        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("최적화를 보고 싶은 지표", options=metric_candidates, index=0)
            st.caption("※ 학교=EC 조건이므로, 4개 점(EC 1/2/4/8) 기반으로 곡선을 ‘참고용’으로 적합합니다.")

            df_use = df_growth_all.dropna(subset=[metric, "EC조건"]).copy()
            if df_use.empty:
                st.error("선택한 지표에 대한 유효 데이터가 없습니다.")
            else:
                # EC별 평균/표준편차
                grp = df_use.groupby(["학교", "EC조건"])[metric].agg(["mean", "std", "count"]).reset_index()
                grp = grp.sort_values("EC조건")

                st.write("### EC별 평균 비교")
                st.dataframe(
                    grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"}),
                    use_container_width=True,
                )

                # 생중량이면 하늘고 최적 강조 문구
                if metric == "생중량(g)":
                    st.info("하늘고(EC 2.0)가 **최적** 조건으로 제시되어 있으므로 그래프에서 강조 표시합니다.")

                # 곡선 적합(2차) + 최적점(-b/2a)
                xs = grp["EC조건"].astype(float).values
                ys = grp["mean"].astype(float).values

                predicted_opt = None
                fit_ok = False

                if len(np.unique(xs)) >= 3 and len(xs) >= 3:
                    try:
                        coef = np.polyfit(xs, ys, deg=2)  # a, b, c
                        a, b, c = coef[0], coef[1], coef[2]
                        if a != 0:
                            x_opt = -b / (2 * a)
                            predicted_opt = float(x_opt)
                            fit_ok = True
                    except Exception:
                        fit_ok = False

                # 그래프
                fig = go.Figure()

                # 점(평균) + 에러바
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        name="EC별 평균",
                        error_y=dict(
                            type="data",
                            array=np.nan_to_num(grp["std"].values, nan=0.0),
                            visible=True,
                        ),
                        hovertemplate="EC=%{x}<br>평균=%{y:.4f}<extra></extra>",
                    )
                )

                # 하늘고(EC 2.0) 강조 마커(겹쳐서 표시)
                if 2.0 in xs:
                    y_2 = float(grp.loc[grp["EC조건"] == 2.0, "mean"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[2.0],
                            y=[y_2],
                            mode="markers",
                            name="하늘고(EC 2.0) 강조",
                            marker=dict(size=16, symbol="star"),
                            hovertemplate="하늘고(EC 2.0)<br>평균=%{y:.4f}<extra></extra>",
                        )
                    )

                # 적합 곡선
                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="2차 적합(참고)",
                            hoverinfo="skip",
                        )
                    )

                    # 예측 최적 EC 표시(범위 내로만 안내)
                    fig.add_vline(
                        x=predicted_opt,
                        line_dash="dash",
                        annotation_text="예측 최적 EC(참고)",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    xaxis_title="EC",
                    yaxis_title=metric + " (평균)",
                    margin=dict(l=30, r=30, t=60, b=30),
                    height=600,
                )
                fig = plotly_apply_font(fig)
                st.plotly_chart(fig, use_container_width=True)

                # 해석 요약
                st.write("### 해석 요약")
                if fit_ok and predicted_opt is not None:
                    st.write(
                        "- 2차 적합 기반 예측 최적 EC(참고): **"
                        + f"{predicted_opt:.2f}"
                        + "**"
                    )
                    st.write("- 단, 학교 4개 점으로 적합한 결과라 **참고용**이며, 반복 실험/중간 EC(예: 1.5, 2.5, 3.0 등) 검증이 필요합니다.")
                else:
                    st.write("- 현재 데이터로는 안정적인 곡선 적합이 어려워(값 부족/오류) 예측 최적 EC를 계산하지 않았습니다.")

                # 다운로드: 요약 + (선택) 예측 곡선 점
                st.subheader("분석 결과 다운로드")
                sheets = {}
                sheets["EC별요약"] = grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"})

                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    df_curve = pd.DataFrame({"EC": x_line, "예측값(2차적합)": y_line})
                    sheets["2차적합곡선"] = df_curve

                buffer = to_excel_bytes_multi(sheets)
                st.download_button(
                    label="최적 EC 분석 XLSX 다운로드",
                    data=buffer,
                    file_name="최적EC_분석결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="극지식물 생육 최적 EC 탐구 대시보드",
    layout="wide",
)

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

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_OPTIONS = ["전체"] + SCHOOLS

EC_CONDITION = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(강조)
    "아라고": 4.0,
    "동산고": 8.0,
}


# -----------------------------
# 유틸: 한글 NFC/NFD 안전 비교
# -----------------------------
def _norm_set(text: str) -> set:
    if text is None:
        return set()
    return {
        unicodedata.normalize("NFC", str(text)),
        unicodedata.normalize("NFD", str(text)),
    }


def _token_in_text(text: str, token: str) -> bool:
    text_variants = _norm_set(text)
    token_variants = _norm_set(token)
    for tv in text_variants:
        for kv in token_variants:
            if kv in tv:
                return True
    return False


def _best_school_match(name_or_sheet: str) -> str | None:
    # 파일명/시트명에서 학교명을 "포함 여부"로 찾기 (NFC/NFD 양방향)
    for sch in SCHOOLS:
        if _token_in_text(name_or_sheet, sch):
            return sch
    return None


def _safe_sheet_name(name: str) -> str:
    # 엑셀 시트명 제한(31자) + 기본 정리
    base = unicodedata.normalize("NFC", str(name)).strip()
    if not base:
        base = "data"
    return base[:31]


# -----------------------------
# 파일 탐색 (iterdir + NFC/NFD)
# -----------------------------
@st.cache_data(show_spinner=False)
def discover_assets(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    env_csv_by_school: dict[str, str] = {}
    xlsx_path: str = ""

    if not data_dir.exists():
        return {"env_csv_by_school": {}, "growth_xlsx": ""}

    csv_candidates = []
    xlsx_candidates = []

    for p in data_dir.iterdir():  # ✅ 필수: iterdir()
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            csv_candidates.append(p)
        elif suf in [".xlsx", ".xlsm", ".xls"]:
            xlsx_candidates.append(p)

    # 환경 CSV: 학교명 포함 + "환경" "데이터" 토큰 포함 (하드코딩된 전체 파일명 사용 X)
    for p in csv_candidates:
        nm = p.name
        if _token_in_text(nm, "환경") and _token_in_text(nm, "데이터"):
            sch = _best_school_match(nm)
            if sch is None:
                # 혹시 "학교명_" 형태가 아니더라도 대응 (스플릿은 보조적으로만 사용)
                stem = unicodedata.normalize("NFC", p.stem)
                parts = stem.split("_")
                if parts:
                    candidate = parts[0].strip()
                    # candidate가 실제 학교 옵션 중 하나면 채택
                    for s in SCHOOLS:
                        if _token_in_text(candidate, s) or _token_in_text(s, candidate):
                            sch = s
                            break
            if sch is not None:
                env_csv_by_school[sch] = str(p)

    # 생육결과 XLSX: "생육" + "결과" 또는 "생육" + "데이터" 같은 토큰 기반 탐지
    for p in xlsx_candidates:
        nm = p.name
        if (_token_in_text(nm, "생육") and _token_in_text(nm, "결과")) or (
            _token_in_text(nm, "생육") and _token_in_text(nm, "데이터")
        ):
            xlsx_path = str(p)
            break
    if (not xlsx_path) and xlsx_candidates:
        # 토큰 매칭이 실패했더라도 xlsx가 1개뿐인 상황을 대비한 안전 fallback
        xlsx_path = str(xlsx_candidates[0])

    return {"env_csv_by_school": env_csv_by_school, "growth_xlsx": xlsx_path}


# -----------------------------
# 데이터 로딩 (cache + 안전 인코딩)
# -----------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    # 학교별 CSV 인코딩이 다를 수 있어 utf-8-sig -> cp949 순으로 시도
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막: pandas 기본
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_env_csv(path: str, school: str) -> pd.DataFrame:
    df = _read_csv_safely(path)

    # 컬럼 표준화
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["time", "temperature", "humidity", "ph", "ec"]
    for c in needed:
        if c not in df.columns:
            # 일부 파일에서 대문자/다른 표기일 수 있어 유사 검색
            for original in df.columns:
                if _token_in_text(original, c):
                    df = df.rename(columns={original: c})
                    break

    missing = [c for c in needed if c not in df.columns]
    if missing:
        # 필수 컬럼이 없으면 빈 DF 반환 (상위에서 st.error)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df["학교"] = school
    df["EC조건"] = EC_CONDITION.get(school, np.nan)
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    frames = []
    for sheet in xl.sheet_names:  # ✅ 시트명 하드코딩 금지: 실제 sheet_names 사용
        sch = _best_school_match(sheet)
        if sch is None:
            continue

        try:
            df = xl.parse(sheet_name=sheet)
        except Exception:
            continue

        # 컬럼명 트림
        df.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df.columns]

        # 기대 컬럼(한글) 기반으로 숫자형 변환
        numeric_like = []
        for c in df.columns:
            if c == "개체번호":
                continue
            numeric_like.append(c)

        for c in numeric_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["학교"] = sch
        df["EC조건"] = EC_CONDITION.get(sch, np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# -----------------------------
# 다운로드 유틸 (BytesIO + openpyxl)
# -----------------------------
def to_excel_bytes_single(df: pd.DataFrame, sheet_name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    safe = _safe_sheet_name(sheet_name)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


def to_excel_bytes_multi(sheets: dict[str, pd.DataFrame]) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = _safe_sheet_name(name)
            df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


# -----------------------------
# 분석 유틸
# -----------------------------
def env_summary(df_env_all: pd.DataFrame) -> pd.DataFrame:
    if df_env_all.empty:
        return pd.DataFrame()

    g = df_env_all.groupby("학교", dropna=False)
    out = g.agg(
        측정수=("time", "count"),
        평균온도=("temperature", "mean"),
        평균습도=("humidity", "mean"),
        평균pH=("ph", "mean"),
        평균EC=("ec", "mean"),
    ).reset_index()

    for c in out.columns:
        if c != "학교" and c != "측정수":
            out[c] = out[c].round(3)
    return out


def growth_summary(df_growth_all: pd.DataFrame) -> pd.DataFrame:
    if df_growth_all.empty:
        return pd.DataFrame()

    # 존재하는 지표만 요약
    candidates = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    metrics = [c for c in candidates if c in df_growth_all.columns]

    agg_dict = {}
    for m in metrics:
        agg_dict[m + "_평균"] = (m, "mean")
        agg_dict[m + "_표준편차"] = (m, "std")

    out = df_growth_all.groupby(["학교", "EC조건"], dropna=False).agg(**agg_dict).reset_index()

    # 반올림
    for c in out.columns:
        if c not in ["학교", "EC조건"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out


def plotly_apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    return fig


# -----------------------------
# UI 시작
# -----------------------------
st.title("극지식물 생육 최적 EC 탐구 대시보드")

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

with st.spinner("데이터 파일 탐색 중..."):
    assets = discover_assets(str(data_dir))

env_paths = assets.get("env_csv_by_school", {}) or {}
growth_xlsx = assets.get("growth_xlsx", "") or ""

# 사이드바
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", options=SCHOOL_OPTIONS, index=0)

# 데이터 로딩
df_env_all_list = []
if env_paths:
    with st.spinner("환경 데이터(CSV) 로딩 중..."):
        for sch, pth in env_paths.items():
            try:
                dfi = load_env_csv(pth, sch)
                if not dfi.empty:
                    df_env_all_list.append(dfi)
            except Exception:
                # 개별 파일 오류는 전체 중단 대신 경고로 처리
                st.sidebar.warning("환경 데이터 일부를 읽지 못했습니다: " + unicodedata.normalize("NFC", sch))

df_env_all = pd.concat(df_env_all_list, ignore_index=True) if df_env_all_list else pd.DataFrame()

with st.spinner("생육 결과 데이터(XLSX) 로딩 중..."):
    df_growth_all = load_growth_xlsx(growth_xlsx)

# 유효성 안내
if not env_paths:
    st.error("data 폴더에서 환경 CSV 파일을 찾지 못했습니다. (예: '...환경...데이터...csv')")
if not growth_xlsx:
    st.error("data 폴더에서 생육 결과 XLSX 파일을 찾지 못했습니다. (예: '...생육...결과...xlsx')")

if df_env_all.empty:
    st.error("환경 데이터가 비어있거나 필수 컬럼(time, temperature, humidity, ph, ec)을 찾지 못했습니다.")
if df_growth_all.empty:
    st.error("생육 결과 데이터가 비어있거나 시트에서 학교명을 인식하지 못했습니다.")

# 선택 필터
if selected_school != "전체":
    df_env = df_env_all[df_env_all["학교"] == selected_school].copy() if not df_env_all.empty else pd.DataFrame()
    df_growth = df_growth_all[df_growth_all["학교"] == selected_school].copy() if not df_growth_all.empty else pd.DataFrame()
else:
    df_env = df_env_all.copy()
    df_growth = df_growth_all.copy()

tabs = st.tabs(
    [
        "탭1: 극지식물 최적 EC 농도 연구 대시보드",
        "탭2: EC 농도에 따른 극지식물 생육 비교 분석",
        "탭3: 4개교 데이터로 찾는 최적 EC (1.0~8.0)",
    ]
)

# -----------------------------
# 탭1
# -----------------------------
with tabs[0]:
    st.subheader("학교별 생육환경 비교 (환경 데이터)")

    if df_env.empty:
        st.error("선택된 조건에서 표시할 환경 데이터가 없습니다.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("환경 측정 레코드 수", f"{len(df_env):,}")
        colB.metric("평균 온도", f"{df_env['temperature'].mean():.2f}")
        colC.metric("평균 습도", f"{df_env['humidity'].mean():.2f}")
        colD.metric("평균 pH", f"{df_env['ph'].mean():.2f}")

        st.caption("※ 학교별 측정 주기가 달라 시간축 비교 시 주의하세요.")

        # (1) 시계열 서브플롯
        fig_ts = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("온도(temperature)", "습도(humidity)", "pH(ph)", "EC(ec)"),
        )

        for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
            dfi = df_env[df_env["학교"] == sch].copy()
            if dfi.empty:
                continue

            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["temperature"], mode="lines", name=sch), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["humidity"], mode="lines", name=sch, showlegend=False), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ph"], mode="lines", name=sch, showlegend=False), row=3, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ec"], mode="lines", name=sch, showlegend=False), row=4, col=1)

        fig_ts.update_height(900)
        fig_ts.update_layout(
            margin=dict(l=30, r=30, t=70, b=30),
            legend_title_text="학교",
        )
        fig_ts = plotly_apply_font(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

        # (2) 분포 비교(박스플롯) - 학교별 비교가 핵심이므로 전체일 때 가치가 큼
        if selected_school == "전체":
            st.subheader("학교별 환경 분포 비교")
            fig_box = make_subplots(rows=2, cols=2, subplot_titles=("온도", "습도", "pH", "EC"))
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["temperature"], name="온도"), row=1, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["humidity"], name="습도", showlegend=False), row=1, col=2)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ph"], name="pH", showlegend=False), row=2, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ec"], name="EC", showlegend=False), row=2, col=2)
            fig_box.update_height(700)
            fig_box.update_layout(margin=dict(l=30, r=30, t=70, b=30))
            fig_box = plotly_apply_font(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        # (3) 환경 요약표 + 다운로드
        st.subheader("환경 요약(학교별 평균)")
        df_env_sum = env_summary(df_env_all)
        if df_env_sum.empty:
            st.error("환경 요약을 만들 수 없습니다.")
        else:
            st.dataframe(df_env_sum, use_container_width=True)

            buf = to_excel_bytes_single(df_env_sum, "환경요약")
            st.download_button(
                label="환경 요약 XLSX 다운로드",
                data=buf,
                file_name="환경요약.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("원본 환경 데이터 다운로드")
        buf2 = to_excel_bytes_single(df_env, "환경데이터")
        # 파일명 f-string 금지 → 문자열 덧셈 사용
        down_name = "전체_환경데이터.xlsx" if selected_school == "전체" else (selected_school + "_환경데이터.xlsx")
        st.download_button(
            label="선택된 환경 데이터 XLSX 다운로드",
            data=buf2,
            file_name=down_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# -----------------------------
# 탭2
# -----------------------------
with tabs[1]:
    st.subheader("EC 농도에 따른 극지식물 생육 비교 (생육 결과)")

    if df_growth.empty:
        st.error("선택된 조건에서 표시할 생육 결과 데이터가 없습니다.")
    else:
        # 생육 지표 선택
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth.columns]

        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("비교할 생육 지표 선택", options=metric_candidates, index=0)

            # 학교(=EC조건)별 분포
            st.caption("※ 각 학교는 서로 다른 EC 조건(1.0, 2.0, 4.0, 8.0)에서 재배한 결과입니다.")
            df_plot = df_growth.dropna(subset=[metric, "EC조건"]).copy()

            if df_plot.empty:
                st.error("선택한 지표에 대해 유효한 값이 없습니다.")
            else:
                # 박스플롯: x=EC, color/hover=학교
                fig = go.Figure()

                # 하늘고(EC 2.0) 강조: 두꺼운 라인 + 주석
                for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
                    dfi = df_plot[df_plot["학교"] == sch]
                    if dfi.empty:
                        continue

                    is_best = (sch == "하늘고")
                    fig.add_trace(
                        go.Box(
                            x=dfi["EC조건"],
                            y=dfi[metric],
                            name=sch,
                            boxmean="sd",
                            line=dict(width=4 if is_best else 2),
                        )
                    )

                fig.update_layout(
                    xaxis_title="EC 조건",
                    yaxis_title=metric,
                    margin=dict(l=30, r=30, t=60, b=30),
                )
                fig = plotly_apply_font(fig)

                # 하늘고(EC 2.0) 최적 강조: 세로 하이라이트 영역
                fig.add_vrect(
                    x0=1.8, x1=2.2,
                    fillcolor="rgba(0, 0, 0, 0.08)",
                    line_width=0,
                    annotation_text="하늘고(EC 2.0) 최적(강조)",
                    annotation_position="top left",
                )

                st.plotly_chart(fig, use_container_width=True)

                # 요약표
                st.subheader("학교(EC)별 요약 통계")
                df_sum = growth_summary(df_growth_all if selected_school == "전체" else df_growth)
                if df_sum.empty:
                    st.error("요약 통계를 만들 수 없습니다.")
                else:
                    st.dataframe(df_sum, use_container_width=True)

                    buf = to_excel_bytes_single(df_sum, "생육요약")
                    down_name = "전체_생육요약.xlsx" if selected_school == "전체" else (selected_school + "_생육요약.xlsx")
                    st.download_button(
                        label="생육 요약 XLSX 다운로드",
                        data=buf,
                        file_name=down_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # 환경요소 vs 생육(학교 평균 기반 상관) — 표본이 4개(학교 4개)라 참고용임을 명시
                st.subheader("환경 요소 영향(참고): 학교 평균 기반 상관 비교")
                if df_env_all.empty:
                    st.error("환경 데이터가 없어 상관 분석을 진행할 수 없습니다.")
                else:
                    # 학교별 평균 환경 + 학교별 평균 생육(선택 metric)
                    env_avg = df_env_all.groupby("학교")[["temperature", "humidity", "ph", "ec"]].mean().reset_index()
                    grow_avg = df_growth_all.groupby("학교")[[metric]].mean().reset_index()

                    merged = pd.merge(env_avg, grow_avg, on="학교", how="inner")
                    merged = merged.dropna()

                    if len(merged) < 3:
                        st.error("상관을 계산하기에 데이터가 부족합니다.")
                    else:
                        corr_cols = ["temperature", "humidity", "ph", "ec", metric]
                        corr = merged[corr_cols].corr()

                        fig_hm = go.Figure(
                            data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.index,
                                zmin=-1,
                                zmax=1,
                                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
                            )
                        )
                        fig_hm.update_layout(
                            height=520,
                            margin=dict(l=30, r=30, t=30, b=30),
                        )
                        fig_hm = plotly_apply_font(fig_hm)
                        st.plotly_chart(fig_hm, use_container_width=True)

                        # 어떤 요소가 더 "세게" 움직였는지: |corr| 비교 (참고)
                        strength = corr[metric].drop(metric).abs().sort_values(ascending=False)
                        st.write("**(참고) 선택 지표와의 |상관계수| 순위**")
                        st.dataframe(strength.rename("abs(corr)"), use_container_width=True)

                st.subheader("원본 생육 데이터 다운로드")
                buf2 = to_excel_bytes_single(df_growth, "생육데이터")
                down_name2 = "전체_생육데이터.xlsx" if selected_school == "전체" else (selected_school + "_생육데이터.xlsx")
                st.download_button(
                    label="선택된 생육 데이터 XLSX 다운로드",
                    data=buf2,
                    file_name=down_name2,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# -----------------------------
# 탭3
# -----------------------------
with tabs[2]:
    st.subheader("4개교 데이터로 최적 EC 추정 (1.0 ~ 8.0)")

    if df_growth_all.empty:
        st.error("생육 결과 데이터가 없어 최적 EC 추정을 할 수 없습니다.")
    else:
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth_all.columns]
        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("최적화를 보고 싶은 지표", options=metric_candidates, index=0)
            st.caption("※ 학교=EC 조건이므로, 4개 점(EC 1/2/4/8) 기반으로 곡선을 ‘참고용’으로 적합합니다.")

            df_use = df_growth_all.dropna(subset=[metric, "EC조건"]).copy()
            if df_use.empty:
                st.error("선택한 지표에 대한 유효 데이터가 없습니다.")
            else:
                # EC별 평균/표준편차
                grp = df_use.groupby(["학교", "EC조건"])[metric].agg(["mean", "std", "count"]).reset_index()
                grp = grp.sort_values("EC조건")

                st.write("### EC별 평균 비교")
                st.dataframe(
                    grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"}),
                    use_container_width=True,
                )

                # 생중량이면 하늘고 최적 강조 문구
                if metric == "생중량(g)":
                    st.info("하늘고(EC 2.0)가 **최적** 조건으로 제시되어 있으므로 그래프에서 강조 표시합니다.")

                # 곡선 적합(2차) + 최적점(-b/2a)
                xs = grp["EC조건"].astype(float).values
                ys = grp["mean"].astype(float).values

                predicted_opt = None
                fit_ok = False

                if len(np.unique(xs)) >= 3 and len(xs) >= 3:
                    try:
                        coef = np.polyfit(xs, ys, deg=2)  # a, b, c
                        a, b, c = coef[0], coef[1], coef[2]
                        if a != 0:
                            x_opt = -b / (2 * a)
                            predicted_opt = float(x_opt)
                            fit_ok = True
                    except Exception:
                        fit_ok = False

                # 그래프
                fig = go.Figure()

                # 점(평균) + 에러바
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        name="EC별 평균",
                        error_y=dict(
                            type="data",
                            array=np.nan_to_num(grp["std"].values, nan=0.0),
                            visible=True,
                        ),
                        hovertemplate="EC=%{x}<br>평균=%{y:.4f}<extra></extra>",
                    )
                )

                # 하늘고(EC 2.0) 강조 마커(겹쳐서 표시)
                if 2.0 in xs:
                    y_2 = float(grp.loc[grp["EC조건"] == 2.0, "mean"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[2.0],
                            y=[y_2],
                            mode="markers",
                            name="하늘고(EC 2.0) 강조",
                            marker=dict(size=16, symbol="star"),
                            hovertemplate="하늘고(EC 2.0)<br>평균=%{y:.4f}<extra></extra>",
                        )
                    )

                # 적합 곡선
                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="2차 적합(참고)",
                            hoverinfo="skip",
                        )
                    )

                    # 예측 최적 EC 표시(범위 내로만 안내)
                    fig.add_vline(
                        x=predicted_opt,
                        line_dash="dash",
                        annotation_text="예측 최적 EC(참고)",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    xaxis_title="EC",
                    yaxis_title=metric + " (평균)",
                    margin=dict(l=30, r=30, t=60, b=30),
                    height=600,
                )
                fig = plotly_apply_font(fig)
                st.plotly_chart(fig, use_container_width=True)

                # 해석 요약
                st.write("### 해석 요약")
                if fit_ok and predicted_opt is not None:
                    st.write(
                        "- 2차 적합 기반 예측 최적 EC(참고): **"
                        + f"{predicted_opt:.2f}"
                        + "**"
                    )
                    st.write("- 단, 학교 4개 점으로 적합한 결과라 **참고용**이며, 반복 실험/중간 EC(예: 1.5, 2.5, 3.0 등) 검증이 필요합니다.")
                else:
                    st.write("- 현재 데이터로는 안정적인 곡선 적합이 어려워(값 부족/오류) 예측 최적 EC를 계산하지 않았습니다.")

                # 다운로드: 요약 + (선택) 예측 곡선 점
                st.subheader("분석 결과 다운로드")
                sheets = {}
                sheets["EC별요약"] = grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"})

                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    df_curve = pd.DataFrame({"EC": x_line, "예측값(2차적합)": y_line})
                    sheets["2차적합곡선"] = df_curve

                buffer = to_excel_bytes_multi(sheets)
                st.download_button(
                    label="최적 EC 분석 XLSX 다운로드",
                    data=buffer,
                    file_name="최적EC_분석결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="극지식물 생육 최적 EC 탐구 대시보드",
    layout="wide",
)

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

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_OPTIONS = ["전체"] + SCHOOLS

EC_CONDITION = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(강조)
    "아라고": 4.0,
    "동산고": 8.0,
}


# -----------------------------
# 유틸: 한글 NFC/NFD 안전 비교
# -----------------------------
def _norm_set(text: str) -> set:
    if text is None:
        return set()
    return {
        unicodedata.normalize("NFC", str(text)),
        unicodedata.normalize("NFD", str(text)),
    }


def _token_in_text(text: str, token: str) -> bool:
    text_variants = _norm_set(text)
    token_variants = _norm_set(token)
    for tv in text_variants:
        for kv in token_variants:
            if kv in tv:
                return True
    return False


def _best_school_match(name_or_sheet: str) -> str | None:
    # 파일명/시트명에서 학교명을 "포함 여부"로 찾기 (NFC/NFD 양방향)
    for sch in SCHOOLS:
        if _token_in_text(name_or_sheet, sch):
            return sch
    return None


def _safe_sheet_name(name: str) -> str:
    # 엑셀 시트명 제한(31자) + 기본 정리
    base = unicodedata.normalize("NFC", str(name)).strip()
    if not base:
        base = "data"
    return base[:31]


# -----------------------------
# 파일 탐색 (iterdir + NFC/NFD)
# -----------------------------
@st.cache_data(show_spinner=False)
def discover_assets(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    env_csv_by_school: dict[str, str] = {}
    xlsx_path: str = ""

    if not data_dir.exists():
        return {"env_csv_by_school": {}, "growth_xlsx": ""}

    csv_candidates = []
    xlsx_candidates = []

    for p in data_dir.iterdir():  # ✅ 필수: iterdir()
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            csv_candidates.append(p)
        elif suf in [".xlsx", ".xlsm", ".xls"]:
            xlsx_candidates.append(p)

    # 환경 CSV: 학교명 포함 + "환경" "데이터" 토큰 포함 (하드코딩된 전체 파일명 사용 X)
    for p in csv_candidates:
        nm = p.name
        if _token_in_text(nm, "환경") and _token_in_text(nm, "데이터"):
            sch = _best_school_match(nm)
            if sch is None:
                # 혹시 "학교명_" 형태가 아니더라도 대응 (스플릿은 보조적으로만 사용)
                stem = unicodedata.normalize("NFC", p.stem)
                parts = stem.split("_")
                if parts:
                    candidate = parts[0].strip()
                    # candidate가 실제 학교 옵션 중 하나면 채택
                    for s in SCHOOLS:
                        if _token_in_text(candidate, s) or _token_in_text(s, candidate):
                            sch = s
                            break
            if sch is not None:
                env_csv_by_school[sch] = str(p)

    # 생육결과 XLSX: "생육" + "결과" 또는 "생육" + "데이터" 같은 토큰 기반 탐지
    for p in xlsx_candidates:
        nm = p.name
        if (_token_in_text(nm, "생육") and _token_in_text(nm, "결과")) or (
            _token_in_text(nm, "생육") and _token_in_text(nm, "데이터")
        ):
            xlsx_path = str(p)
            break
    if (not xlsx_path) and xlsx_candidates:
        # 토큰 매칭이 실패했더라도 xlsx가 1개뿐인 상황을 대비한 안전 fallback
        xlsx_path = str(xlsx_candidates[0])

    return {"env_csv_by_school": env_csv_by_school, "growth_xlsx": xlsx_path}


# -----------------------------
# 데이터 로딩 (cache + 안전 인코딩)
# -----------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    # 학교별 CSV 인코딩이 다를 수 있어 utf-8-sig -> cp949 순으로 시도
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막: pandas 기본
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_env_csv(path: str, school: str) -> pd.DataFrame:
    df = _read_csv_safely(path)

    # 컬럼 표준화
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["time", "temperature", "humidity", "ph", "ec"]
    for c in needed:
        if c not in df.columns:
            # 일부 파일에서 대문자/다른 표기일 수 있어 유사 검색
            for original in df.columns:
                if _token_in_text(original, c):
                    df = df.rename(columns={original: c})
                    break

    missing = [c for c in needed if c not in df.columns]
    if missing:
        # 필수 컬럼이 없으면 빈 DF 반환 (상위에서 st.error)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df["학교"] = school
    df["EC조건"] = EC_CONDITION.get(school, np.nan)
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    frames = []
    for sheet in xl.sheet_names:  # ✅ 시트명 하드코딩 금지: 실제 sheet_names 사용
        sch = _best_school_match(sheet)
        if sch is None:
            continue

        try:
            df = xl.parse(sheet_name=sheet)
        except Exception:
            continue

        # 컬럼명 트림
        df.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df.columns]

        # 기대 컬럼(한글) 기반으로 숫자형 변환
        numeric_like = []
        for c in df.columns:
            if c == "개체번호":
                continue
            numeric_like.append(c)

        for c in numeric_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["학교"] = sch
        df["EC조건"] = EC_CONDITION.get(sch, np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# -----------------------------
# 다운로드 유틸 (BytesIO + openpyxl)
# -----------------------------
def to_excel_bytes_single(df: pd.DataFrame, sheet_name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    safe = _safe_sheet_name(sheet_name)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


def to_excel_bytes_multi(sheets: dict[str, pd.DataFrame]) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = _safe_sheet_name(name)
            df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


# -----------------------------
# 분석 유틸
# -----------------------------
def env_summary(df_env_all: pd.DataFrame) -> pd.DataFrame:
    if df_env_all.empty:
        return pd.DataFrame()

    g = df_env_all.groupby("학교", dropna=False)
    out = g.agg(
        측정수=("time", "count"),
        평균온도=("temperature", "mean"),
        평균습도=("humidity", "mean"),
        평균pH=("ph", "mean"),
        평균EC=("ec", "mean"),
    ).reset_index()

    for c in out.columns:
        if c != "학교" and c != "측정수":
            out[c] = out[c].round(3)
    return out


def growth_summary(df_growth_all: pd.DataFrame) -> pd.DataFrame:
    if df_growth_all.empty:
        return pd.DataFrame()

    # 존재하는 지표만 요약
    candidates = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    metrics = [c for c in candidates if c in df_growth_all.columns]

    agg_dict = {}
    for m in metrics:
        agg_dict[m + "_평균"] = (m, "mean")
        agg_dict[m + "_표준편차"] = (m, "std")

    out = df_growth_all.groupby(["학교", "EC조건"], dropna=False).agg(**agg_dict).reset_index()

    # 반올림
    for c in out.columns:
        if c not in ["학교", "EC조건"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out


def plotly_apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    return fig


# -----------------------------
# UI 시작
# -----------------------------
st.title("극지식물 생육 최적 EC 탐구 대시보드")

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

with st.spinner("데이터 파일 탐색 중..."):
    assets = discover_assets(str(data_dir))

env_paths = assets.get("env_csv_by_school", {}) or {}
growth_xlsx = assets.get("growth_xlsx", "") or ""

# 사이드바
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", options=SCHOOL_OPTIONS, index=0)

# 데이터 로딩
df_env_all_list = []
if env_paths:
    with st.spinner("환경 데이터(CSV) 로딩 중..."):
        for sch, pth in env_paths.items():
            try:
                dfi = load_env_csv(pth, sch)
                if not dfi.empty:
                    df_env_all_list.append(dfi)
            except Exception:
                # 개별 파일 오류는 전체 중단 대신 경고로 처리
                st.sidebar.warning("환경 데이터 일부를 읽지 못했습니다: " + unicodedata.normalize("NFC", sch))

df_env_all = pd.concat(df_env_all_list, ignore_index=True) if df_env_all_list else pd.DataFrame()

with st.spinner("생육 결과 데이터(XLSX) 로딩 중..."):
    df_growth_all = load_growth_xlsx(growth_xlsx)

# 유효성 안내
if not env_paths:
    st.error("data 폴더에서 환경 CSV 파일을 찾지 못했습니다. (예: '...환경...데이터...csv')")
if not growth_xlsx:
    st.error("data 폴더에서 생육 결과 XLSX 파일을 찾지 못했습니다. (예: '...생육...결과...xlsx')")

if df_env_all.empty:
    st.error("환경 데이터가 비어있거나 필수 컬럼(time, temperature, humidity, ph, ec)을 찾지 못했습니다.")
if df_growth_all.empty:
    st.error("생육 결과 데이터가 비어있거나 시트에서 학교명을 인식하지 못했습니다.")

# 선택 필터
if selected_school != "전체":
    df_env = df_env_all[df_env_all["학교"] == selected_school].copy() if not df_env_all.empty else pd.DataFrame()
    df_growth = df_growth_all[df_growth_all["학교"] == selected_school].copy() if not df_growth_all.empty else pd.DataFrame()
else:
    df_env = df_env_all.copy()
    df_growth = df_growth_all.copy()

tabs = st.tabs(
    [
        "탭1: 극지식물 최적 EC 농도 연구 대시보드",
        "탭2: EC 농도에 따른 극지식물 생육 비교 분석",
        "탭3: 4개교 데이터로 찾는 최적 EC (1.0~8.0)",
    ]
)

# -----------------------------
# 탭1
# -----------------------------
with tabs[0]:
    st.subheader("학교별 생육환경 비교 (환경 데이터)")

    if df_env.empty:
        st.error("선택된 조건에서 표시할 환경 데이터가 없습니다.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("환경 측정 레코드 수", f"{len(df_env):,}")
        colB.metric("평균 온도", f"{df_env['temperature'].mean():.2f}")
        colC.metric("평균 습도", f"{df_env['humidity'].mean():.2f}")
        colD.metric("평균 pH", f"{df_env['ph'].mean():.2f}")

        st.caption("※ 학교별 측정 주기가 달라 시간축 비교 시 주의하세요.")

        # (1) 시계열 서브플롯
        fig_ts = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("온도(temperature)", "습도(humidity)", "pH(ph)", "EC(ec)"),
        )

        for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
            dfi = df_env[df_env["학교"] == sch].copy()
            if dfi.empty:
                continue

            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["temperature"], mode="lines", name=sch), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["humidity"], mode="lines", name=sch, showlegend=False), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ph"], mode="lines", name=sch, showlegend=False), row=3, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ec"], mode="lines", name=sch, showlegend=False), row=4, col=1)

        fig_ts.update_height(900)
        fig_ts.update_layout(
            margin=dict(l=30, r=30, t=70, b=30),
            legend_title_text="학교",
        )
        fig_ts = plotly_apply_font(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

        # (2) 분포 비교(박스플롯) - 학교별 비교가 핵심이므로 전체일 때 가치가 큼
        if selected_school == "전체":
            st.subheader("학교별 환경 분포 비교")
            fig_box = make_subplots(rows=2, cols=2, subplot_titles=("온도", "습도", "pH", "EC"))
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["temperature"], name="온도"), row=1, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["humidity"], name="습도", showlegend=False), row=1, col=2)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ph"], name="pH", showlegend=False), row=2, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ec"], name="EC", showlegend=False), row=2, col=2)
            fig_box.update_height(700)
            fig_box.update_layout(margin=dict(l=30, r=30, t=70, b=30))
            fig_box = plotly_apply_font(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        # (3) 환경 요약표 + 다운로드
        st.subheader("환경 요약(학교별 평균)")
        df_env_sum = env_summary(df_env_all)
        if df_env_sum.empty:
            st.error("환경 요약을 만들 수 없습니다.")
        else:
            st.dataframe(df_env_sum, use_container_width=True)

            buf = to_excel_bytes_single(df_env_sum, "환경요약")
            st.download_button(
                label="환경 요약 XLSX 다운로드",
                data=buf,
                file_name="환경요약.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("원본 환경 데이터 다운로드")
        buf2 = to_excel_bytes_single(df_env, "환경데이터")
        # 파일명 f-string 금지 → 문자열 덧셈 사용
        down_name = "전체_환경데이터.xlsx" if selected_school == "전체" else (selected_school + "_환경데이터.xlsx")
        st.download_button(
            label="선택된 환경 데이터 XLSX 다운로드",
            data=buf2,
            file_name=down_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# -----------------------------
# 탭2
# -----------------------------
with tabs[1]:
    st.subheader("EC 농도에 따른 극지식물 생육 비교 (생육 결과)")

    if df_growth.empty:
        st.error("선택된 조건에서 표시할 생육 결과 데이터가 없습니다.")
    else:
        # 생육 지표 선택
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth.columns]

        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("비교할 생육 지표 선택", options=metric_candidates, index=0)

            # 학교(=EC조건)별 분포
            st.caption("※ 각 학교는 서로 다른 EC 조건(1.0, 2.0, 4.0, 8.0)에서 재배한 결과입니다.")
            df_plot = df_growth.dropna(subset=[metric, "EC조건"]).copy()

            if df_plot.empty:
                st.error("선택한 지표에 대해 유효한 값이 없습니다.")
            else:
                # 박스플롯: x=EC, color/hover=학교
                fig = go.Figure()

                # 하늘고(EC 2.0) 강조: 두꺼운 라인 + 주석
                for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
                    dfi = df_plot[df_plot["학교"] == sch]
                    if dfi.empty:
                        continue

                    is_best = (sch == "하늘고")
                    fig.add_trace(
                        go.Box(
                            x=dfi["EC조건"],
                            y=dfi[metric],
                            name=sch,
                            boxmean="sd",
                            line=dict(width=4 if is_best else 2),
                        )
                    )

                fig.update_layout(
                    xaxis_title="EC 조건",
                    yaxis_title=metric,
                    margin=dict(l=30, r=30, t=60, b=30),
                )
                fig = plotly_apply_font(fig)

                # 하늘고(EC 2.0) 최적 강조: 세로 하이라이트 영역
                fig.add_vrect(
                    x0=1.8, x1=2.2,
                    fillcolor="rgba(0, 0, 0, 0.08)",
                    line_width=0,
                    annotation_text="하늘고(EC 2.0) 최적(강조)",
                    annotation_position="top left",
                )

                st.plotly_chart(fig, use_container_width=True)

                # 요약표
                st.subheader("학교(EC)별 요약 통계")
                df_sum = growth_summary(df_growth_all if selected_school == "전체" else df_growth)
                if df_sum.empty:
                    st.error("요약 통계를 만들 수 없습니다.")
                else:
                    st.dataframe(df_sum, use_container_width=True)

                    buf = to_excel_bytes_single(df_sum, "생육요약")
                    down_name = "전체_생육요약.xlsx" if selected_school == "전체" else (selected_school + "_생육요약.xlsx")
                    st.download_button(
                        label="생육 요약 XLSX 다운로드",
                        data=buf,
                        file_name=down_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # 환경요소 vs 생육(학교 평균 기반 상관) — 표본이 4개(학교 4개)라 참고용임을 명시
                st.subheader("환경 요소 영향(참고): 학교 평균 기반 상관 비교")
                if df_env_all.empty:
                    st.error("환경 데이터가 없어 상관 분석을 진행할 수 없습니다.")
                else:
                    # 학교별 평균 환경 + 학교별 평균 생육(선택 metric)
                    env_avg = df_env_all.groupby("학교")[["temperature", "humidity", "ph", "ec"]].mean().reset_index()
                    grow_avg = df_growth_all.groupby("학교")[[metric]].mean().reset_index()

                    merged = pd.merge(env_avg, grow_avg, on="학교", how="inner")
                    merged = merged.dropna()

                    if len(merged) < 3:
                        st.error("상관을 계산하기에 데이터가 부족합니다.")
                    else:
                        corr_cols = ["temperature", "humidity", "ph", "ec", metric]
                        corr = merged[corr_cols].corr()

                        fig_hm = go.Figure(
                            data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.index,
                                zmin=-1,
                                zmax=1,
                                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
                            )
                        )
                        fig_hm.update_layout(
                            height=520,
                            margin=dict(l=30, r=30, t=30, b=30),
                        )
                        fig_hm = plotly_apply_font(fig_hm)
                        st.plotly_chart(fig_hm, use_container_width=True)

                        # 어떤 요소가 더 "세게" 움직였는지: |corr| 비교 (참고)
                        strength = corr[metric].drop(metric).abs().sort_values(ascending=False)
                        st.write("**(참고) 선택 지표와의 |상관계수| 순위**")
                        st.dataframe(strength.rename("abs(corr)"), use_container_width=True)

                st.subheader("원본 생육 데이터 다운로드")
                buf2 = to_excel_bytes_single(df_growth, "생육데이터")
                down_name2 = "전체_생육데이터.xlsx" if selected_school == "전체" else (selected_school + "_생육데이터.xlsx")
                st.download_button(
                    label="선택된 생육 데이터 XLSX 다운로드",
                    data=buf2,
                    file_name=down_name2,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# -----------------------------
# 탭3
# -----------------------------
with tabs[2]:
    st.subheader("4개교 데이터로 최적 EC 추정 (1.0 ~ 8.0)")

    if df_growth_all.empty:
        st.error("생육 결과 데이터가 없어 최적 EC 추정을 할 수 없습니다.")
    else:
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth_all.columns]
        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("최적화를 보고 싶은 지표", options=metric_candidates, index=0)
            st.caption("※ 학교=EC 조건이므로, 4개 점(EC 1/2/4/8) 기반으로 곡선을 ‘참고용’으로 적합합니다.")

            df_use = df_growth_all.dropna(subset=[metric, "EC조건"]).copy()
            if df_use.empty:
                st.error("선택한 지표에 대한 유효 데이터가 없습니다.")
            else:
                # EC별 평균/표준편차
                grp = df_use.groupby(["학교", "EC조건"])[metric].agg(["mean", "std", "count"]).reset_index()
                grp = grp.sort_values("EC조건")

                st.write("### EC별 평균 비교")
                st.dataframe(
                    grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"}),
                    use_container_width=True,
                )

                # 생중량이면 하늘고 최적 강조 문구
                if metric == "생중량(g)":
                    st.info("하늘고(EC 2.0)가 **최적** 조건으로 제시되어 있으므로 그래프에서 강조 표시합니다.")

                # 곡선 적합(2차) + 최적점(-b/2a)
                xs = grp["EC조건"].astype(float).values
                ys = grp["mean"].astype(float).values

                predicted_opt = None
                fit_ok = False

                if len(np.unique(xs)) >= 3 and len(xs) >= 3:
                    try:
                        coef = np.polyfit(xs, ys, deg=2)  # a, b, c
                        a, b, c = coef[0], coef[1], coef[2]
                        if a != 0:
                            x_opt = -b / (2 * a)
                            predicted_opt = float(x_opt)
                            fit_ok = True
                    except Exception:
                        fit_ok = False

                # 그래프
                fig = go.Figure()

                # 점(평균) + 에러바
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        name="EC별 평균",
                        error_y=dict(
                            type="data",
                            array=np.nan_to_num(grp["std"].values, nan=0.0),
                            visible=True,
                        ),
                        hovertemplate="EC=%{x}<br>평균=%{y:.4f}<extra></extra>",
                    )
                )

                # 하늘고(EC 2.0) 강조 마커(겹쳐서 표시)
                if 2.0 in xs:
                    y_2 = float(grp.loc[grp["EC조건"] == 2.0, "mean"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[2.0],
                            y=[y_2],
                            mode="markers",
                            name="하늘고(EC 2.0) 강조",
                            marker=dict(size=16, symbol="star"),
                            hovertemplate="하늘고(EC 2.0)<br>평균=%{y:.4f}<extra></extra>",
                        )
                    )

                # 적합 곡선
                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="2차 적합(참고)",
                            hoverinfo="skip",
                        )
                    )

                    # 예측 최적 EC 표시(범위 내로만 안내)
                    fig.add_vline(
                        x=predicted_opt,
                        line_dash="dash",
                        annotation_text="예측 최적 EC(참고)",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    xaxis_title="EC",
                    yaxis_title=metric + " (평균)",
                    margin=dict(l=30, r=30, t=60, b=30),
                    height=600,
                )
                fig = plotly_apply_font(fig)
                st.plotly_chart(fig, use_container_width=True)

                # 해석 요약
                st.write("### 해석 요약")
                if fit_ok and predicted_opt is not None:
                    st.write(
                        "- 2차 적합 기반 예측 최적 EC(참고): **"
                        + f"{predicted_opt:.2f}"
                        + "**"
                    )
                    st.write("- 단, 학교 4개 점으로 적합한 결과라 **참고용**이며, 반복 실험/중간 EC(예: 1.5, 2.5, 3.0 등) 검증이 필요합니다.")
                else:
                    st.write("- 현재 데이터로는 안정적인 곡선 적합이 어려워(값 부족/오류) 예측 최적 EC를 계산하지 않았습니다.")

                # 다운로드: 요약 + (선택) 예측 곡선 점
                st.subheader("분석 결과 다운로드")
                sheets = {}
                sheets["EC별요약"] = grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"})

                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    df_curve = pd.DataFrame({"EC": x_line, "예측값(2차적합)": y_line})
                    sheets["2차적합곡선"] = df_curve

                buffer = to_excel_bytes_multi(sheets)
                st.download_button(
                    label="최적 EC 분석 XLSX 다운로드",
                    data=buffer,
                    file_name="최적EC_분석결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="극지식물 생육 최적 EC 탐구 대시보드",
    layout="wide",
)

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

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_OPTIONS = ["전체"] + SCHOOLS

EC_CONDITION = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(강조)
    "아라고": 4.0,
    "동산고": 8.0,
}


# -----------------------------
# 유틸: 한글 NFC/NFD 안전 비교
# -----------------------------
def _norm_set(text: str) -> set:
    if text is None:
        return set()
    return {
        unicodedata.normalize("NFC", str(text)),
        unicodedata.normalize("NFD", str(text)),
    }


def _token_in_text(text: str, token: str) -> bool:
    text_variants = _norm_set(text)
    token_variants = _norm_set(token)
    for tv in text_variants:
        for kv in token_variants:
            if kv in tv:
                return True
    return False


def _best_school_match(name_or_sheet: str) -> str | None:
    # 파일명/시트명에서 학교명을 "포함 여부"로 찾기 (NFC/NFD 양방향)
    for sch in SCHOOLS:
        if _token_in_text(name_or_sheet, sch):
            return sch
    return None


def _safe_sheet_name(name: str) -> str:
    # 엑셀 시트명 제한(31자) + 기본 정리
    base = unicodedata.normalize("NFC", str(name)).strip()
    if not base:
        base = "data"
    return base[:31]


# -----------------------------
# 파일 탐색 (iterdir + NFC/NFD)
# -----------------------------
@st.cache_data(show_spinner=False)
def discover_assets(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    env_csv_by_school: dict[str, str] = {}
    xlsx_path: str = ""

    if not data_dir.exists():
        return {"env_csv_by_school": {}, "growth_xlsx": ""}

    csv_candidates = []
    xlsx_candidates = []

    for p in data_dir.iterdir():  # ✅ 필수: iterdir()
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            csv_candidates.append(p)
        elif suf in [".xlsx", ".xlsm", ".xls"]:
            xlsx_candidates.append(p)

    # 환경 CSV: 학교명 포함 + "환경" "데이터" 토큰 포함 (하드코딩된 전체 파일명 사용 X)
    for p in csv_candidates:
        nm = p.name
        if _token_in_text(nm, "환경") and _token_in_text(nm, "데이터"):
            sch = _best_school_match(nm)
            if sch is None:
                # 혹시 "학교명_" 형태가 아니더라도 대응 (스플릿은 보조적으로만 사용)
                stem = unicodedata.normalize("NFC", p.stem)
                parts = stem.split("_")
                if parts:
                    candidate = parts[0].strip()
                    # candidate가 실제 학교 옵션 중 하나면 채택
                    for s in SCHOOLS:
                        if _token_in_text(candidate, s) or _token_in_text(s, candidate):
                            sch = s
                            break
            if sch is not None:
                env_csv_by_school[sch] = str(p)

    # 생육결과 XLSX: "생육" + "결과" 또는 "생육" + "데이터" 같은 토큰 기반 탐지
    for p in xlsx_candidates:
        nm = p.name
        if (_token_in_text(nm, "생육") and _token_in_text(nm, "결과")) or (
            _token_in_text(nm, "생육") and _token_in_text(nm, "데이터")
        ):
            xlsx_path = str(p)
            break
    if (not xlsx_path) and xlsx_candidates:
        # 토큰 매칭이 실패했더라도 xlsx가 1개뿐인 상황을 대비한 안전 fallback
        xlsx_path = str(xlsx_candidates[0])

    return {"env_csv_by_school": env_csv_by_school, "growth_xlsx": xlsx_path}


# -----------------------------
# 데이터 로딩 (cache + 안전 인코딩)
# -----------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    # 학교별 CSV 인코딩이 다를 수 있어 utf-8-sig -> cp949 순으로 시도
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막: pandas 기본
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_env_csv(path: str, school: str) -> pd.DataFrame:
    df = _read_csv_safely(path)

    # 컬럼 표준화
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["time", "temperature", "humidity", "ph", "ec"]
    for c in needed:
        if c not in df.columns:
            # 일부 파일에서 대문자/다른 표기일 수 있어 유사 검색
            for original in df.columns:
                if _token_in_text(original, c):
                    df = df.rename(columns={original: c})
                    break

    missing = [c for c in needed if c not in df.columns]
    if missing:
        # 필수 컬럼이 없으면 빈 DF 반환 (상위에서 st.error)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df["학교"] = school
    df["EC조건"] = EC_CONDITION.get(school, np.nan)
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    frames = []
    for sheet in xl.sheet_names:  # ✅ 시트명 하드코딩 금지: 실제 sheet_names 사용
        sch = _best_school_match(sheet)
        if sch is None:
            continue

        try:
            df = xl.parse(sheet_name=sheet)
        except Exception:
            continue

        # 컬럼명 트림
        df.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df.columns]

        # 기대 컬럼(한글) 기반으로 숫자형 변환
        numeric_like = []
        for c in df.columns:
            if c == "개체번호":
                continue
            numeric_like.append(c)

        for c in numeric_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["학교"] = sch
        df["EC조건"] = EC_CONDITION.get(sch, np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# -----------------------------
# 다운로드 유틸 (BytesIO + openpyxl)
# -----------------------------
def to_excel_bytes_single(df: pd.DataFrame, sheet_name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    safe = _safe_sheet_name(sheet_name)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


def to_excel_bytes_multi(sheets: dict[str, pd.DataFrame]) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = _safe_sheet_name(name)
            df.to_excel(writer, index=False, sheet_name=safe)
    buffer.seek(0)
    return buffer


# -----------------------------
# 분석 유틸
# -----------------------------
def env_summary(df_env_all: pd.DataFrame) -> pd.DataFrame:
    if df_env_all.empty:
        return pd.DataFrame()

    g = df_env_all.groupby("학교", dropna=False)
    out = g.agg(
        측정수=("time", "count"),
        평균온도=("temperature", "mean"),
        평균습도=("humidity", "mean"),
        평균pH=("ph", "mean"),
        평균EC=("ec", "mean"),
    ).reset_index()

    for c in out.columns:
        if c != "학교" and c != "측정수":
            out[c] = out[c].round(3)
    return out


def growth_summary(df_growth_all: pd.DataFrame) -> pd.DataFrame:
    if df_growth_all.empty:
        return pd.DataFrame()

    # 존재하는 지표만 요약
    candidates = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    metrics = [c for c in candidates if c in df_growth_all.columns]

    agg_dict = {}
    for m in metrics:
        agg_dict[m + "_평균"] = (m, "mean")
        agg_dict[m + "_표준편차"] = (m, "std")

    out = df_growth_all.groupby(["학교", "EC조건"], dropna=False).agg(**agg_dict).reset_index()

    # 반올림
    for c in out.columns:
        if c not in ["학교", "EC조건"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out


def plotly_apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    return fig


# -----------------------------
# UI 시작
# -----------------------------
st.title("극지식물 생육 최적 EC 탐구 대시보드")

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

with st.spinner("데이터 파일 탐색 중..."):
    assets = discover_assets(str(data_dir))

env_paths = assets.get("env_csv_by_school", {}) or {}
growth_xlsx = assets.get("growth_xlsx", "") or ""

# 사이드바
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", options=SCHOOL_OPTIONS, index=0)

# 데이터 로딩
df_env_all_list = []
if env_paths:
    with st.spinner("환경 데이터(CSV) 로딩 중..."):
        for sch, pth in env_paths.items():
            try:
                dfi = load_env_csv(pth, sch)
                if not dfi.empty:
                    df_env_all_list.append(dfi)
            except Exception:
                # 개별 파일 오류는 전체 중단 대신 경고로 처리
                st.sidebar.warning("환경 데이터 일부를 읽지 못했습니다: " + unicodedata.normalize("NFC", sch))

df_env_all = pd.concat(df_env_all_list, ignore_index=True) if df_env_all_list else pd.DataFrame()

with st.spinner("생육 결과 데이터(XLSX) 로딩 중..."):
    df_growth_all = load_growth_xlsx(growth_xlsx)

# 유효성 안내
if not env_paths:
    st.error("data 폴더에서 환경 CSV 파일을 찾지 못했습니다. (예: '...환경...데이터...csv')")
if not growth_xlsx:
    st.error("data 폴더에서 생육 결과 XLSX 파일을 찾지 못했습니다. (예: '...생육...결과...xlsx')")

if df_env_all.empty:
    st.error("환경 데이터가 비어있거나 필수 컬럼(time, temperature, humidity, ph, ec)을 찾지 못했습니다.")
if df_growth_all.empty:
    st.error("생육 결과 데이터가 비어있거나 시트에서 학교명을 인식하지 못했습니다.")

# 선택 필터
if selected_school != "전체":
    df_env = df_env_all[df_env_all["학교"] == selected_school].copy() if not df_env_all.empty else pd.DataFrame()
    df_growth = df_growth_all[df_growth_all["학교"] == selected_school].copy() if not df_growth_all.empty else pd.DataFrame()
else:
    df_env = df_env_all.copy()
    df_growth = df_growth_all.copy()

tabs = st.tabs(
    [
        "탭1: 극지식물 최적 EC 농도 연구 대시보드",
        "탭2: EC 농도에 따른 극지식물 생육 비교 분석",
        "탭3: 4개교 데이터로 찾는 최적 EC (1.0~8.0)",
    ]
)

# -----------------------------
# 탭1
# -----------------------------
with tabs[0]:
    st.subheader("학교별 생육환경 비교 (환경 데이터)")

    if df_env.empty:
        st.error("선택된 조건에서 표시할 환경 데이터가 없습니다.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("환경 측정 레코드 수", f"{len(df_env):,}")
        colB.metric("평균 온도", f"{df_env['temperature'].mean():.2f}")
        colC.metric("평균 습도", f"{df_env['humidity'].mean():.2f}")
        colD.metric("평균 pH", f"{df_env['ph'].mean():.2f}")

        st.caption("※ 학교별 측정 주기가 달라 시간축 비교 시 주의하세요.")

        # (1) 시계열 서브플롯
        fig_ts = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("온도(temperature)", "습도(humidity)", "pH(ph)", "EC(ec)"),
        )

        for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
            dfi = df_env[df_env["학교"] == sch].copy()
            if dfi.empty:
                continue

            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["temperature"], mode="lines", name=sch), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["humidity"], mode="lines", name=sch, showlegend=False), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ph"], mode="lines", name=sch, showlegend=False), row=3, col=1)
            fig_ts.add_trace(go.Scatter(x=dfi["time"], y=dfi["ec"], mode="lines", name=sch, showlegend=False), row=4, col=1)

        fig_ts.update_height(900)
        fig_ts.update_layout(
            margin=dict(l=30, r=30, t=70, b=30),
            legend_title_text="학교",
        )
        fig_ts = plotly_apply_font(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

        # (2) 분포 비교(박스플롯) - 학교별 비교가 핵심이므로 전체일 때 가치가 큼
        if selected_school == "전체":
            st.subheader("학교별 환경 분포 비교")
            fig_box = make_subplots(rows=2, cols=2, subplot_titles=("온도", "습도", "pH", "EC"))
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["temperature"], name="온도"), row=1, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["humidity"], name="습도", showlegend=False), row=1, col=2)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ph"], name="pH", showlegend=False), row=2, col=1)
            fig_box.add_trace(go.Box(x=df_env["학교"], y=df_env["ec"], name="EC", showlegend=False), row=2, col=2)
            fig_box.update_height(700)
            fig_box.update_layout(margin=dict(l=30, r=30, t=70, b=30))
            fig_box = plotly_apply_font(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        # (3) 환경 요약표 + 다운로드
        st.subheader("환경 요약(학교별 평균)")
        df_env_sum = env_summary(df_env_all)
        if df_env_sum.empty:
            st.error("환경 요약을 만들 수 없습니다.")
        else:
            st.dataframe(df_env_sum, use_container_width=True)

            buf = to_excel_bytes_single(df_env_sum, "환경요약")
            st.download_button(
                label="환경 요약 XLSX 다운로드",
                data=buf,
                file_name="환경요약.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("원본 환경 데이터 다운로드")
        buf2 = to_excel_bytes_single(df_env, "환경데이터")
        # 파일명 f-string 금지 → 문자열 덧셈 사용
        down_name = "전체_환경데이터.xlsx" if selected_school == "전체" else (selected_school + "_환경데이터.xlsx")
        st.download_button(
            label="선택된 환경 데이터 XLSX 다운로드",
            data=buf2,
            file_name=down_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# -----------------------------
# 탭2
# -----------------------------
with tabs[1]:
    st.subheader("EC 농도에 따른 극지식물 생육 비교 (생육 결과)")

    if df_growth.empty:
        st.error("선택된 조건에서 표시할 생육 결과 데이터가 없습니다.")
    else:
        # 생육 지표 선택
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth.columns]

        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("비교할 생육 지표 선택", options=metric_candidates, index=0)

            # 학교(=EC조건)별 분포
            st.caption("※ 각 학교는 서로 다른 EC 조건(1.0, 2.0, 4.0, 8.0)에서 재배한 결과입니다.")
            df_plot = df_growth.dropna(subset=[metric, "EC조건"]).copy()

            if df_plot.empty:
                st.error("선택한 지표에 대해 유효한 값이 없습니다.")
            else:
                # 박스플롯: x=EC, color/hover=학교
                fig = go.Figure()

                # 하늘고(EC 2.0) 강조: 두꺼운 라인 + 주석
                for sch in (SCHOOLS if selected_school == "전체" else [selected_school]):
                    dfi = df_plot[df_plot["학교"] == sch]
                    if dfi.empty:
                        continue

                    is_best = (sch == "하늘고")
                    fig.add_trace(
                        go.Box(
                            x=dfi["EC조건"],
                            y=dfi[metric],
                            name=sch,
                            boxmean="sd",
                            line=dict(width=4 if is_best else 2),
                        )
                    )

                fig.update_layout(
                    xaxis_title="EC 조건",
                    yaxis_title=metric,
                    margin=dict(l=30, r=30, t=60, b=30),
                )
                fig = plotly_apply_font(fig)

                # 하늘고(EC 2.0) 최적 강조: 세로 하이라이트 영역
                fig.add_vrect(
                    x0=1.8, x1=2.2,
                    fillcolor="rgba(0, 0, 0, 0.08)",
                    line_width=0,
                    annotation_text="하늘고(EC 2.0) 최적(강조)",
                    annotation_position="top left",
                )

                st.plotly_chart(fig, use_container_width=True)

                # 요약표
                st.subheader("학교(EC)별 요약 통계")
                df_sum = growth_summary(df_growth_all if selected_school == "전체" else df_growth)
                if df_sum.empty:
                    st.error("요약 통계를 만들 수 없습니다.")
                else:
                    st.dataframe(df_sum, use_container_width=True)

                    buf = to_excel_bytes_single(df_sum, "생육요약")
                    down_name = "전체_생육요약.xlsx" if selected_school == "전체" else (selected_school + "_생육요약.xlsx")
                    st.download_button(
                        label="생육 요약 XLSX 다운로드",
                        data=buf,
                        file_name=down_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # 환경요소 vs 생육(학교 평균 기반 상관) — 표본이 4개(학교 4개)라 참고용임을 명시
                st.subheader("환경 요소 영향(참고): 학교 평균 기반 상관 비교")
                if df_env_all.empty:
                    st.error("환경 데이터가 없어 상관 분석을 진행할 수 없습니다.")
                else:
                    # 학교별 평균 환경 + 학교별 평균 생육(선택 metric)
                    env_avg = df_env_all.groupby("학교")[["temperature", "humidity", "ph", "ec"]].mean().reset_index()
                    grow_avg = df_growth_all.groupby("학교")[[metric]].mean().reset_index()

                    merged = pd.merge(env_avg, grow_avg, on="학교", how="inner")
                    merged = merged.dropna()

                    if len(merged) < 3:
                        st.error("상관을 계산하기에 데이터가 부족합니다.")
                    else:
                        corr_cols = ["temperature", "humidity", "ph", "ec", metric]
                        corr = merged[corr_cols].corr()

                        fig_hm = go.Figure(
                            data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.index,
                                zmin=-1,
                                zmax=1,
                                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
                            )
                        )
                        fig_hm.update_layout(
                            height=520,
                            margin=dict(l=30, r=30, t=30, b=30),
                        )
                        fig_hm = plotly_apply_font(fig_hm)
                        st.plotly_chart(fig_hm, use_container_width=True)

                        # 어떤 요소가 더 "세게" 움직였는지: |corr| 비교 (참고)
                        strength = corr[metric].drop(metric).abs().sort_values(ascending=False)
                        st.write("**(참고) 선택 지표와의 |상관계수| 순위**")
                        st.dataframe(strength.rename("abs(corr)"), use_container_width=True)

                st.subheader("원본 생육 데이터 다운로드")
                buf2 = to_excel_bytes_single(df_growth, "생육데이터")
                down_name2 = "전체_생육데이터.xlsx" if selected_school == "전체" else (selected_school + "_생육데이터.xlsx")
                st.download_button(
                    label="선택된 생육 데이터 XLSX 다운로드",
                    data=buf2,
                    file_name=down_name2,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# -----------------------------
# 탭3
# -----------------------------
with tabs[2]:
    st.subheader("4개교 데이터로 최적 EC 추정 (1.0 ~ 8.0)")

    if df_growth_all.empty:
        st.error("생육 결과 데이터가 없어 최적 EC 추정을 할 수 없습니다.")
    else:
        metric_candidates = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)"]
        metric_candidates = [m for m in metric_candidates if m in df_growth_all.columns]
        if not metric_candidates:
            st.error("생육 결과에서 분석 가능한 지표 컬럼을 찾지 못했습니다.")
        else:
            metric = st.selectbox("최적화를 보고 싶은 지표", options=metric_candidates, index=0)
            st.caption("※ 학교=EC 조건이므로, 4개 점(EC 1/2/4/8) 기반으로 곡선을 ‘참고용’으로 적합합니다.")

            df_use = df_growth_all.dropna(subset=[metric, "EC조건"]).copy()
            if df_use.empty:
                st.error("선택한 지표에 대한 유효 데이터가 없습니다.")
            else:
                # EC별 평균/표준편차
                grp = df_use.groupby(["학교", "EC조건"])[metric].agg(["mean", "std", "count"]).reset_index()
                grp = grp.sort_values("EC조건")

                st.write("### EC별 평균 비교")
                st.dataframe(
                    grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"}),
                    use_container_width=True,
                )

                # 생중량이면 하늘고 최적 강조 문구
                if metric == "생중량(g)":
                    st.info("하늘고(EC 2.0)가 **최적** 조건으로 제시되어 있으므로 그래프에서 강조 표시합니다.")

                # 곡선 적합(2차) + 최적점(-b/2a)
                xs = grp["EC조건"].astype(float).values
                ys = grp["mean"].astype(float).values

                predicted_opt = None
                fit_ok = False

                if len(np.unique(xs)) >= 3 and len(xs) >= 3:
                    try:
                        coef = np.polyfit(xs, ys, deg=2)  # a, b, c
                        a, b, c = coef[0], coef[1], coef[2]
                        if a != 0:
                            x_opt = -b / (2 * a)
                            predicted_opt = float(x_opt)
                            fit_ok = True
                    except Exception:
                        fit_ok = False

                # 그래프
                fig = go.Figure()

                # 점(평균) + 에러바
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        name="EC별 평균",
                        error_y=dict(
                            type="data",
                            array=np.nan_to_num(grp["std"].values, nan=0.0),
                            visible=True,
                        ),
                        hovertemplate="EC=%{x}<br>평균=%{y:.4f}<extra></extra>",
                    )
                )

                # 하늘고(EC 2.0) 강조 마커(겹쳐서 표시)
                if 2.0 in xs:
                    y_2 = float(grp.loc[grp["EC조건"] == 2.0, "mean"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[2.0],
                            y=[y_2],
                            mode="markers",
                            name="하늘고(EC 2.0) 강조",
                            marker=dict(size=16, symbol="star"),
                            hovertemplate="하늘고(EC 2.0)<br>평균=%{y:.4f}<extra></extra>",
                        )
                    )

                # 적합 곡선
                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="2차 적합(참고)",
                            hoverinfo="skip",
                        )
                    )

                    # 예측 최적 EC 표시(범위 내로만 안내)
                    fig.add_vline(
                        x=predicted_opt,
                        line_dash="dash",
                        annotation_text="예측 최적 EC(참고)",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    xaxis_title="EC",
                    yaxis_title=metric + " (평균)",
                    margin=dict(l=30, r=30, t=60, b=30),
                    height=600,
                )
                fig = plotly_apply_font(fig)
                st.plotly_chart(fig, use_container_width=True)

                # 해석 요약
                st.write("### 해석 요약")
                if fit_ok and predicted_opt is not None:
                    st.write(
                        "- 2차 적합 기반 예측 최적 EC(참고): **"
                        + f"{predicted_opt:.2f}"
                        + "**"
                    )
                    st.write("- 단, 학교 4개 점으로 적합한 결과라 **참고용**이며, 반복 실험/중간 EC(예: 1.5, 2.5, 3.0 등) 검증이 필요합니다.")
                else:
                    st.write("- 현재 데이터로는 안정적인 곡선 적합이 어려워(값 부족/오류) 예측 최적 EC를 계산하지 않았습니다.")

                # 다운로드: 요약 + (선택) 예측 곡선 점
                st.subheader("분석 결과 다운로드")
                sheets = {}
                sheets["EC별요약"] = grp.rename(columns={"mean": "평균", "std": "표준편차", "count": "표본수"})

                if fit_ok:
                    x_line = np.linspace(1.0, 8.0, 200)
                    y_line = coef[0] * x_line**2 + coef[1] * x_line + coef[2]
                    df_curve = pd.DataFrame({"EC": x_line, "예측값(2차적합)": y_line})
                    sheets["2차적합곡선"] = df_curve

                buffer = to_excel_bytes_multi(sheets)
                st.download_button(
                    label="최적 EC 분석 XLSX 다운로드",
                    data=buffer,
                    file_name="최적EC_분석결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
