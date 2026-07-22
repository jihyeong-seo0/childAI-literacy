import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="전국 청소년 AI 리터러시 단계구분도",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 전국 청소년 AI 리터러시 단계구분도")
st.caption("시도별 청소년 AI 리터러시 수준을 비교합니다.")


# =========================================================
# 파일 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_NAME = (
    "(데이터) 청소년의 생성형 AI 이용실태 및 "
    "리터러시 증진방안 연구.csv"
)

DATA_PATH = BASE_DIR / DATA_NAME

# 같은 저장소에 sido_kr.geojson을 넣으면 로컬 파일을 우선 사용
GEOJSON_PATH = BASE_DIR / "sido_kr.geojson"

GEOJSON_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/boundaries/"
    "sido_kr.geojson"
)


# =========================================================
# AI 리터러시 문항
# =========================================================
LITERACY_GROUPS = {
    "정보 점검": [
        "Q14_1_1",
        "Q14_1_2",
        "Q14_1_3",
    ],
    "AI 소통": [
        "Q14_2_4",
        "Q14_2_5",
        "Q14_2_6",
    ],
    "창의적 활용": [
        "Q14_3_7",
        "Q14_3_8",
        "Q14_3_9",
    ],
    "윤리적 활용": [
        "Q14_4_10",
        "Q14_4_11",
        "Q14_4_12",
    ],
}


# =========================================================
# DM6 → 시도 코드 및 이름
# =========================================================
DM6_TO_REGION = {
    1: ("11", "서울"),
    2: ("26", "부산"),
    3: ("27", "대구"),
    4: ("28", "인천"),
    5: ("29", "광주"),
    6: ("30", "대전"),
    7: ("31", "울산"),
    8: ("36", "세종"),
    9: ("41", "경기"),
    10: ("51", "강원"),
    11: ("43", "충북"),
    12: ("44", "충남"),
    13: ("52", "전북"),
    14: ("46", "전남"),
    15: ("47", "경북"),
    16: ("48", "경남"),
    17: ("50", "제주"),
}


# =========================================================
# 데이터 읽기
# =========================================================
@st.cache_data
def load_csv(path_string):
    """
    여러 인코딩을 순서대로 시도해 CSV를 읽습니다.
    """

    encodings = [
        "utf-8-sig",
        "cp949",
        "utf-8",
    ]

    last_error = None

    for encoding in encodings:
        try:
            data = pd.read_csv(
                path_string,
                encoding=encoding,
                low_memory=False,
            )

            data.columns = (
                data.columns
                .astype(str)
                .str.strip()
            )

            return data

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"CSV 인코딩을 확인할 수 없습니다: {last_error}"
    )


# =========================================================
# GeoJSON 읽기
# =========================================================
@st.cache_data(ttl=86400)
def load_geojson(local_path_string, url):
    """
    로컬 GeoJSON이 있으면 우선 사용합니다.

    로컬 파일이 없으면 GitHub에서 내려받습니다.
    """

    local_path = Path(local_path_string)

    if local_path.exists():
        with open(
            local_path,
            "r",
            encoding="utf-8",
        ) as file:
            geojson = json.load(file)

        source = "저장소의 sido_kr.geojson"

    else:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
            },
        )

        with urlopen(
            request,
            timeout=15,
        ) as response:
            geojson = json.loads(
                response.read().decode("utf-8")
            )

        source = "GitHub 원격 GeoJSON"

    # GeoJSON 코드가 반드시 2자리 문자열이 되도록 정리
    for feature in geojson.get("features", []):
        properties = feature.setdefault(
            "properties",
            {},
        )

        code = properties.get("코드")

        if code is not None:
            properties["코드"] = str(code).strip().zfill(2)

    return geojson, source


# =========================================================
# AI 리터러시 점수 계산
# =========================================================
def calculate_literacy_scores(
    data,
    columns,
    minimum_answers,
):
    """
    응답자별 AI 리터러시 평균 점수를 계산합니다.
    """

    answers = (
        data[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # 1~5점 범위만 정상 응답으로 사용
    answers = answers.where(
        (answers >= 1)
        & (answers <= 5)
    )

    answer_count = (
        answers
        .notna()
        .sum(axis=1)
    )

    average_scores = answers.mean(
        axis=1,
        skipna=True,
    )

    return average_scores.where(
        answer_count >= minimum_answers
    )


# =========================================================
# 가중치
# =========================================================
def get_weights(data):
    """
    wgt_b가 있으면 가중치를 사용합니다.
    """

    if "wgt_b" in data.columns:
        weights = pd.to_numeric(
            data["wgt_b"],
            errors="coerce",
        )

        weights = weights.where(
            weights > 0
        )

        if weights.notna().any():
            return (
                weights,
                "표준화 가중치 wgt_b 적용",
            )

    return (
        pd.Series(
            1.0,
            index=data.index,
        ),
        "가중치 없음",
    )


# =========================================================
# 시도별 집계
# =========================================================
def aggregate_by_region(
    region_codes,
    scores,
    weights,
    threshold,
):
    working = pd.DataFrame(
        {
            "코드": region_codes,
            "점수": scores,
            "가중치": weights,
        }
    )

    working = working.dropna(
        subset=[
            "코드",
            "점수",
            "가중치",
        ]
    ).copy()

    working = working[
        working["가중치"] > 0
    ]

    if working.empty:
        return pd.DataFrame()

    working["고리터러시가중치"] = (
        working["점수"]
        .ge(threshold)
        .astype(float)
        * working["가중치"]
    )

    working["점수가중합"] = (
        working["점수"]
        * working["가중치"]
    )

    result = (
        working
        .groupby(
            "코드",
            as_index=False,
        )
        .agg(
            응답자수=("점수", "size"),
            가중치합=("가중치", "sum"),
            고리터러시가중치=(
                "고리터러시가중치",
                "sum",
            ),
            점수가중합=("점수가중합", "sum"),
        )
    )

    result["AI 리터러시 비율(%)"] = (
        result["고리터러시가중치"]
        / result["가중치합"]
        * 100
    )

    result["평균 리터러시 점수"] = (
        result["점수가중합"]
        / result["가중치합"]
    )

    return result


# =========================================================
# 데이터 파일 확인 및 읽기
# =========================================================
if not DATA_PATH.exists():
    st.error(
        f"데이터 파일을 찾을 수 없습니다: {DATA_NAME}"
    )

    st.code(
        f"""프로젝트폴더/
├── main.py
├── requirements.txt
└── {DATA_NAME}""",
        language="text",
    )

    st.stop()


try:
    with st.spinner("설문 데이터를 읽고 있습니다..."):
        survey = load_csv(
            str(DATA_PATH)
        )

except Exception as error:
    st.error("데이터 파일을 읽는 중 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


st.success(
    f"데이터를 읽었습니다: "
    f"{survey.shape[0]:,}명 × {survey.shape[1]:,}개 변수"
)


# =========================================================
# 필수 변수 확인
# =========================================================
all_q14_columns = [
    column
    for columns in LITERACY_GROUPS.values()
    for column in columns
]

required_columns = [
    "DM6",
    *all_q14_columns,
]

missing_columns = [
    column
    for column in required_columns
    if column not in survey.columns
]

if missing_columns:
    st.error("필요한 변수를 찾지 못했습니다.")

    st.code(
        "\n".join(missing_columns),
        language="text",
    )

    st.stop()


# =========================================================
# 지도 설정
# =========================================================
st.sidebar.header("지도 설정")

selected_groups = st.sidebar.multiselect(
    "AI 리터러시 영역",
    options=list(LITERACY_GROUPS.keys()),
    default=list(LITERACY_GROUPS.keys()),
)

if not selected_groups:
    st.warning(
        "AI 리터러시 영역을 하나 이상 선택하세요."
    )
    st.stop()


selected_columns = [
    column
    for group in selected_groups
    for column in LITERACY_GROUPS[group]
]


threshold = st.sidebar.slider(
    "높은 AI 리터러시 기준",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1,
)


minimum_answers = st.sidebar.slider(
    "개인별 최소 응답 문항 수",
    min_value=1,
    max_value=len(selected_columns),
    value=max(
        1,
        int(len(selected_columns) * 0.75),
    ),
)


# =========================================================
# 지역 코드 만들기
# =========================================================
dm6 = pd.to_numeric(
    survey["DM6"],
    errors="coerce",
).astype("Int64")


code_mapping = {
    dm6_code: region_code
    for dm6_code, (region_code, region_name)
    in DM6_TO_REGION.items()
}


name_mapping = {
    dm6_code: region_name
    for dm6_code, (region_code, region_name)
    in DM6_TO_REGION.items()
}


survey["코드"] = dm6.map(code_mapping)
survey["시도"] = dm6.map(name_mapping)


# =========================================================
# 점수 계산
# =========================================================
literacy_scores = calculate_literacy_scores(
    survey,
    selected_columns,
    minimum_answers,
)

weights, weight_note = get_weights(survey)


regional_result = aggregate_by_region(
    survey["코드"],
    literacy_scores,
    weights,
    threshold,
)

if regional_result.empty:
    st.error(
        "계산 가능한 AI 리터러시 응답이 없습니다."
    )
    st.stop()


# =========================================================
# 지역 이름 결합
# =========================================================
region_table = pd.DataFrame(
    [
        {
            "코드": region_code,
            "시도": region_name,
        }
        for region_code, region_name
        in DM6_TO_REGION.values()
    ]
)


map_data = region_table.merge(
    regional_result,
    on="코드",
    how="left",
)


# =========================================================
# 전국 통계 계산
# =========================================================
valid = (
    literacy_scores.notna()
    & weights.notna()
    & weights.gt(0)
)


if not valid.any():
    st.error(
        "유효한 AI 리터러시 응답자가 없습니다."
    )
    st.stop()


valid_scores = literacy_scores[valid]
valid_weights = weights[valid]

total_weight = valid_weights.sum()

national_average = (
    valid_scores
    .mul(valid_weights)
    .sum()
    / total_weight
)

national_rate = (
    valid_scores
    .ge(threshold)
    .astype(float)
    .mul(valid_weights)
    .sum()
    / total_weight
    * 100
)


# =========================================================
# 핵심 결과
# =========================================================
metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "전체 응답자",
        f"{len(survey):,}명",
    )

with metric2:
    st.metric(
        "유효 Q14 응답자",
        f"{valid.sum():,}명",
    )

with metric3:
    st.metric(
        "전국 평균 점수",
        f"{national_average:.2f} / 5",
    )

with metric4:
    st.metric(
        "높은 리터러시 비율",
        f"{national_rate:.1f}%",
    )


st.caption(
    f"데이터: {DATA_NAME} · {weight_note}"
)


# =========================================================
# 시도별 결과표
# 지도보다 먼저 표시하므로 지도 오류가 나도 결과 확인 가능
# =========================================================
st.subheader("시도별 AI 리터러시 결과")

ranking = map_data.sort_values(
    "AI 리터러시 비율(%)",
    ascending=False,
).copy()

ranking["평균 리터러시 점수"] = (
    ranking["평균 리터러시 점수"]
    .round(2)
)

ranking["AI 리터러시 비율(%)"] = (
    ranking["AI 리터러시 비율(%)"]
    .round(1)
)

st.dataframe(
    ranking[
        [
            "시도",
            "코드",
            "응답자수",
            "평균 리터러시 점수",
            "AI 리터러시 비율(%)",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 지도
# =========================================================
st.subheader("전국 단계구분도")

try:
    with st.spinner(
        "전국 시도 경계를 불러오고 있습니다..."
    ):
        geojson, geojson_source = load_geojson(
            str(GEOJSON_PATH),
            GEOJSON_URL,
        )

    chart_data = map_data.dropna(
        subset=["AI 리터러시 비율(%)"]
    ).copy()

    figure = px.choropleth(
        chart_data,
        geojson=geojson,
        locations="코드",
        featureidkey="properties.코드",
        color="AI 리터러시 비율(%)",
        hover_name="시도",
        hover_data={
            "코드": False,
            "응답자수": True,
            "평균 리터러시 점수": ":.2f",
            "AI 리터러시 비율(%)": ":.1f",
        },
        color_continuous_scale="Blues",
        range_color=(0, 100),
        labels={
            "AI 리터러시 비율(%)": "AI 리터러시 비율",
            "응답자수": "유효 응답자",
            "평균 리터러시 점수": "평균 점수",
        },
    )

    figure.update_traces(
        marker_line_color="white",
        marker_line_width=0.9,
    )

    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )

    figure.update_layout(
        height=760,
        margin={
            "l": 0,
            "r": 0,
            "t": 20,
            "b": 0,
        },
        coloraxis_colorbar={
            "title": "비율(%)",
            "ticksuffix": "%",
        },
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.caption(
        f"경계 파일: {geojson_source}"
    )

except Exception as error:
    st.error(
        "시도별 계산은 완료됐지만 "
        "지도 경계를 불러오거나 그리는 중 오류가 발생했습니다."
    )

    st.exception(error)

    st.info(
        "가장 안정적으로 실행하려면 "
        "`sido_kr.geojson` 파일을 GitHub 저장소의 "
        "`main.py`와 같은 폴더에 넣으세요."
    )
