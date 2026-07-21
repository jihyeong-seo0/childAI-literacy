import json
import re
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="전국 청소년 AI 리터러시 단계구분도",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# 파일 및 URL 설정
# =========================================================
CODEBOOK_NAME = (
    "(코드북) 청소년의 생성형 AI 이용실태 및 "
    "리터러시 증진방안 연구.csv"
)

RAW_DATA_NAME = (
    "청소년의 생성형 AI 이용실태 및 "
    "리터러시 증진방안 연구.csv"
)

BASE_DIR = Path(__file__).resolve().parent

CODEBOOK_PATH = BASE_DIR / CODEBOOK_NAME
RAW_DATA_PATH = BASE_DIR / RAW_DATA_NAME


SIGUNGU_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/boundaries/"
    "sigungu_kr.geojson"
)

SIDO_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/boundaries/"
    "sido_kr.geojson"
)


# =========================================================
# 코드북 Q14 AI 리터러시 문항
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

ALL_Q14_COLUMNS = [
    column
    for columns in LITERACY_GROUPS.values()
    for column in columns
]


# =========================================================
# 코드북 DM6 → 행정구역 시도 코드
# =========================================================
DM6_TO_SIDO_CODE = {
    1: "11",    # 서울
    2: "26",    # 부산
    3: "27",    # 대구
    4: "28",    # 인천
    5: "29",    # 광주
    6: "30",    # 대전
    7: "31",    # 울산
    8: "36",    # 세종
    9: "41",    # 경기
    10: "51",   # 강원
    11: "43",   # 충북
    12: "44",   # 충남
    13: "52",   # 전북
    14: "46",   # 전남
    15: "47",   # 경북
    16: "48",   # 경남
    17: "50",   # 제주
}


# =========================================================
# CSV 읽기
# =========================================================
def read_csv_flexible(source):
    """
    UTF-8-SIG, CP949, UTF-8 순서로 CSV를 읽습니다.
    """

    encodings = (
        "utf-8-sig",
        "cp949",
        "utf-8",
    )

    if isinstance(source, (str, Path)):
        for encoding in encodings:
            try:
                return pd.read_csv(
                    source,
                    encoding=encoding,
                    low_memory=False,
                )

            except UnicodeDecodeError:
                continue

    else:
        if hasattr(source, "getvalue"):
            raw_bytes = source.getvalue()
        else:
            raw_bytes = source.read()

        for encoding in encodings:
            try:
                return pd.read_csv(
                    BytesIO(raw_bytes),
                    encoding=encoding,
                    low_memory=False,
                )

            except UnicodeDecodeError:
                continue

    raise ValueError(
        "CSV 파일의 인코딩을 확인할 수 없습니다."
    )


@st.cache_data
def load_codebook(path_string):
    """
    코드북을 읽고 변수명의 빈칸을 앞의 값으로 채웁니다.
    """

    codebook = read_csv_flexible(path_string)

    codebook.columns = (
        codebook.columns
        .astype(str)
        .str.strip()
    )

    required_columns = {
        "변수",
        "변수설명",
        "변수값",
        "레이블",
    }

    if not required_columns.issubset(
        codebook.columns
    ):
        raise ValueError(
            "코드북에 변수, 변수설명, "
            "변수값, 레이블 열이 필요합니다."
        )

    codebook[
        [
            "변수",
            "변수설명",
        ]
    ] = codebook[
        [
            "변수",
            "변수설명",
        ]
    ].ffill()

    return codebook


@st.cache_data
def load_uploaded_data(raw_bytes):
    """
    업로드한 설문 원자료를 읽습니다.
    """

    data = read_csv_flexible(
        BytesIO(raw_bytes)
    )

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    return data


@st.cache_data(ttl=86400)
def load_geojson(url):
    """
    GitHub에서 GeoJSON 경계 파일을 읽습니다.
    """

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:

        text = response.read().decode(
            "utf-8"
        )

    return json.loads(text)


# =========================================================
# 지역 코드 처리
# =========================================================
def normalize_sigungu_code(value):
    """
    시군구 코드를 문자열 5자리로 정리합니다.

    10자리 행정동 코드가 들어오면 앞 5자리를 사용합니다.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    text = re.sub(
        r"\.0$",
        "",
        text,
    )

    digits = re.sub(
        r"[^0-9]",
        "",
        text,
    )

    if len(digits) >= 10:
        return digits[:5]

    if len(digits) == 5:
        return digits

    return None


def find_sigungu_code_column(data):
    """
    원자료에서 시군구 코드로 사용할 열을 찾습니다.
    """

    candidates = [
        "코드",
        "시군구코드",
        "지역코드",
        "행정구역코드",
        "code",
        "CODE",
    ]

    for column in candidates:

        if column not in data.columns:
            continue

        codes = data[column].map(
            normalize_sigungu_code
        )

        if codes.notna().any():
            return column, codes

    empty_codes = pd.Series(
        index=data.index,
        dtype="object",
    )

    return None, empty_codes


# =========================================================
# AI 리터러시 계산
# =========================================================
def calculate_literacy_scores(
    data,
    columns,
    minimum_answer_count,
):
    """
    Q14 문항의 개인별 평균을 계산합니다.
    """

    answers = (
        data[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    answer_count = (
        answers
        .notna()
        .sum(axis=1)
    )

    scores = answers.mean(
        axis=1,
        skipna=True,
    )

    return scores.where(
        answer_count
        >= minimum_answer_count
    )


def get_weights(data):
    """
    wgt_b가 있으면 표준화 가중치를 사용합니다.
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
                "표준화 가중치(wgt_b) 적용",
            )

    default_weights = pd.Series(
        1.0,
        index=data.index,
    )

    return (
        default_weights,
        "가중치 없음",
    )


def aggregate_by_region(
    region_codes,
    literacy_scores,
    weights,
    threshold,
):
    """
    지역별 고리터러시 응답자 비율과
    평균 리터러시 점수를 계산합니다.
    """

    working = pd.DataFrame(
        {
            "코드": region_codes,
            "리터러시점수": literacy_scores,
            "가중치": weights,
        }
    )

    working = working.dropna(
        subset=[
            "코드",
            "리터러시점수",
            "가중치",
        ]
    )

    working = working[
        working["가중치"] > 0
    ].copy()

    if working.empty:
        return pd.DataFrame()

    working[
        "고리터러시가중치"
    ] = (
        working[
            "리터러시점수"
        ].ge(
            threshold
        ).astype(float)
        * working["가중치"]
    )

    working[
        "점수가중합"
    ] = (
        working["리터러시점수"]
        * working["가중치"]
    )

    result = (
        working
        .groupby(
            "코드",
            as_index=False,
        )
        .agg(
            응답자수=(
                "리터러시점수",
                "size",
            ),
            전체가중치=(
                "가중치",
                "sum",
            ),
            고리터러시가중치=(
                "고리터러시가중치",
                "sum",
            ),
            점수가중합=(
                "점수가중합",
                "sum",
            ),
        )
    )

    result[
        "AI 리터러시 비율(%)"
    ] = (
        result["고리터러시가중치"]
        / result["전체가중치"]
        * 100
    )

    result[
        "평균 리터러시 점수"
    ] = (
        result["점수가중합"]
        / result["전체가중치"]
    )

    return result[
        [
            "코드",
            "응답자수",
            "평균 리터러시 점수",
            "AI 리터러시 비율(%)",
        ]
    ]


# =========================================================
# GeoJSON 속성 처리
# =========================================================
def get_boundary_table(
    geojson,
    region_level,
):
    """
    GeoJSON에서 코드와 지역 이름을 가져옵니다.
    """

    if region_level == "시군구":
        name_candidates = [
            "시군구",
            "sggnm",
            "SIG_KOR_NM",
            "name",
        ]

    else:
        name_candidates = [
            "시도",
            "sidonm",
            "CTP_KOR_NM",
            "name",
        ]

    features = geojson.get(
        "features",
        [],
    )

    if not features:
        return pd.DataFrame(
            columns=[
                "코드",
                "지역명",
            ]
        )

    first_properties = (
        features[0]
        .get(
            "properties",
            {},
        )
    )

    name_property = next(
        (
            key
            for key in name_candidates
            if key in first_properties
        ),
        None,
    )

    rows = []

    for feature in features:

        properties = feature.get(
            "properties",
            {},
        )

        code = str(
            properties.get(
                "코드",
                "",
            )
        ).strip()

        name = str(
            properties.get(
                name_property,
                code,
            )
        ).strip()

        rows.append(
            {
                "코드": code,
                "지역명": name,
            }
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=["코드"]
        )
    )


# =========================================================
# Plotly 단계구분도
# =========================================================
def create_choropleth(
    map_data,
    geojson,
    region_level,
    threshold,
):
    """
    배경 타일 없이 행정구역 경계와 색상만 표시합니다.
    """

    colored_data = map_data.dropna(
        subset=[
            "AI 리터러시 비율(%)",
        ]
    ).copy()

    values = colored_data[
        "AI 리터러시 비율(%)"
    ]

    if values.empty:
        color_min = 0.0
        color_max = 100.0

    else:
        color_min = max(
            0.0,
            float(values.min()) - 5,
        )

        color_max = min(
            100.0,
            float(values.max()) + 5,
        )

        if color_max <= color_min:
            color_min = max(
                0.0,
                color_min - 1,
            )

            color_max = min(
                100.0,
                color_max + 1,
            )

    figure = go.Figure()

    # 전국 경계
    figure.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=map_data["코드"],
            z=[0] * len(map_data),
            featureidkey="properties.코드",
            colorscale=[
                [
                    0,
                    "#eef1f5",
                ],
                [
                    1,
                    "#eef1f5",
                ],
            ],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.7,
            hoverinfo="skip",
        )
    )

    # AI 리터러시 비율
    figure.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=colored_data["코드"],
            z=colored_data[
                "AI 리터러시 비율(%)"
            ],
            featureidkey="properties.코드",
            colorscale="Blues",
            zmin=color_min,
            zmax=color_max,
            marker_line_color="white",
            marker_line_width=0.7,
            colorbar={
                "title": {
                    "text": (
                        "고리터러시"
                        "<br>비율(%)"
                    ),
                },
                "ticksuffix": "%",
                "thickness": 16,
                "len": 0.72,
            },
            customdata=colored_data[
                [
                    "지역명",
                    "응답자수",
                    "평균 리터러시 점수",
                ]
            ],
            hovertemplate=(
                f"<b>{region_level}: "
                "%{customdata[0]}</b><br>"
                "AI 리터러시 비율: "
                "%{z:.1f}%<br>"
                "평균 점수: "
                "%{customdata[2]:.2f} / 5<br>"
                "유효 응답자: "
                "%{customdata[1]:,.0f}명"
                "<extra></extra>"
            ),
        )
    )

    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
    )

    figure.update_layout(
        title={
            "text": (
                "전국 청소년 AI 리터러시 "
                "단계구분도"
                "<br>"
                f"<sup>{region_level}별 · "
                f"평균 {threshold:.1f}점 이상 "
                "응답자 비율</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        height=760,
        margin={
            "l": 0,
            "r": 0,
            "t": 80,
            "b": 0,
        },
        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),
    )

    return figure


# =========================================================
# 화면 제목
# =========================================================
st.title(
    "🧠 전국 청소년 AI 리터러시 단계구분도"
)

st.caption(
    "Q14의 정보 점검·AI 소통·창의적 활용·"
    "윤리적 활용 12문항을 이용합니다."
)


with st.expander(
    "AI 리터러시 비율 계산 기준"
):

    st.markdown(
        """
- 응답자별 AI 리터러시 점수: 선택한 Q14 문항의 평균
- 문항 점수 범위: 1점부터 5점
- 고리터러시 응답자: 평균 점수가 설정한 기준 이상인 응답자
- 지역별 AI 리터러시 비율: 고리터러시 응답자의 가중합 ÷ 유효 응답자의 가중합 × 100
- `wgt_b`가 있으면 표준화 가중치를 사용하고, 없으면 단순 비율을 사용
- 인구자료의 `계_` 열과 고령화율은 사용하지 않습니다.
- 이번 비율의 분모는 Q14 문항에 유효하게 답한 설문 응답자입니다.
        """
    )


# =========================================================
# 코드북 읽기
# =========================================================
if not CODEBOOK_PATH.exists():

    st.error(
        "코드북 파일을 찾을 수 없습니다."
    )

    st.code(
        f"""프로젝트폴더/
├── main.py
├── requirements.txt
├── {CODEBOOK_NAME}
└── {RAW_DATA_NAME}""",
        language="text",
    )

    st.stop()


try:
    codebook = load_codebook(
        str(CODEBOOK_PATH)
    )

except Exception as error:

    st.error(
        f"코드북을 읽지 못했습니다: {error}"
    )

    st.stop()


# =========================================================
# 원자료 업로드 또는 로컬 파일 읽기
# =========================================================
st.sidebar.header(
    "데이터 설정"
)

uploaded_file = st.sidebar.file_uploader(
    "학생별 설문 원자료 CSV",
    type=["csv"],
    help=(
        "Q14 문항과 지역 코드가 포함된 "
        "응답자별 원자료를 올리세요."
    ),
)


if uploaded_file is not None:

    st.session_state[
        "survey_raw_bytes"
    ] = uploaded_file.getvalue()

    st.session_state[
        "survey_file_name"
    ] = uploaded_file.name


try:

    if (
        "survey_raw_bytes"
        in st.session_state
    ):

        survey = load_uploaded_data(
            st.session_state[
                "survey_raw_bytes"
            ]
        )

        survey_file_name = (
            st.session_state.get(
                "survey_file_name",
                "업로드 원자료",
            )
        )

    elif RAW_DATA_PATH.exists():

        survey = read_csv_flexible(
            RAW_DATA_PATH
        )

        survey.columns = (
            survey.columns
            .astype(str)
            .str.strip()
        )

        survey_file_name = (
            RAW_DATA_NAME
        )

    else:

        st.info(
            "왼쪽에서 설문 원자료 CSV를 "
            "업로드하거나 GitHub 저장소에 "
            f"`{RAW_DATA_NAME}` 파일을 "
            "넣어주세요."
        )

        st.stop()

except Exception as error:

    st.error(
        f"원자료를 읽지 못했습니다: {error}"
    )

    st.stop()


# =========================================================
# Q14 변수 확인
# =========================================================
missing_q14_columns = [
    column
    for column in ALL_Q14_COLUMNS
    if column not in survey.columns
]


if missing_q14_columns:

    st.error(
        "원자료에서 다음 AI 리터러시 문항을 "
        "찾지 못했습니다: "
        + ", ".join(
            missing_q14_columns
        )
    )

    st.stop()


# =========================================================
# 사이드바 설정
# =========================================================
selected_groups = st.sidebar.multiselect(
    "AI 리터러시 영역",
    options=list(
        LITERACY_GROUPS.keys()
    ),
    default=list(
        LITERACY_GROUPS.keys()
    ),
)


if not selected_groups:

    st.warning(
        "AI 리터러시 영역을 "
        "하나 이상 선택해주세요."
    )

    st.stop()


selected_columns = [
    column
    for group in selected_groups
    for column in LITERACY_GROUPS[group]
]


threshold = st.sidebar.slider(
    "고리터러시 기준 점수",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1,
)


default_minimum_answers = max(
    1,
    int(
        len(selected_columns)
        * 0.75
    ),
)


minimum_answer_count = (
    st.sidebar.slider(
        "최소 응답 문항 수",
        min_value=1,
        max_value=len(
            selected_columns
        ),
        value=default_minimum_answers,
        help=(
            "이 개수 이상의 문항에 "
            "답한 응답자만 평균을 계산합니다."
        ),
    )
)


minimum_region_sample = (
    st.sidebar.number_input(
        "지도에 표시할 최소 지역 표본 수",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
        help=(
            "응답자가 너무 적은 지역은 "
            "색칠하지 않습니다."
        ),
    )
)


# =========================================================
# 지역 코드 및 행정단위 결정
# =========================================================
code_column, sigungu_codes = (
    find_sigungu_code_column(
        survey
    )
)


if code_column is not None:

    region_level = "시군구"

    region_codes = sigungu_codes

    geojson_url = SIGUNGU_URL

    code_note = (
        f"`{code_column}` 열을 "
        "5자리 문자열 코드로 변환"
    )


elif "DM6" in survey.columns:

    region_level = "시도"

    dm6_values = pd.to_numeric(
        survey["DM6"],
        errors="coerce",
    )

    region_codes = dm6_values.map(
        DM6_TO_SIDO_CODE
    )

    geojson_url = SIDO_URL

    code_note = (
        "`DM6` 값을 2자리 "
        "시도 코드로 변환"
    )


else:

    st.error(
        "지역 코드를 찾지 못했습니다. "
        "시군구 지도에는 5자리 `코드` 열이 "
        "필요하며, 시도 지도에는 `DM6` 열이 "
        "필요합니다."
    )

    st.stop()


# =========================================================
# 경계 파일 읽기
# =========================================================
try:

    geojson = load_geojson(
        geojson_url
    )

except Exception as error:

    st.error(
        "경계 GeoJSON을 읽지 못했습니다: "
        f"{error}"
    )

    st.stop()


# =========================================================
# 점수 및 지역별 비율 계산
# =========================================================
literacy_scores = (
    calculate_literacy_scores(
        survey,
        selected_columns,
        minimum_answer_count,
    )
)


weights, weight_note = (
    get_weights(
        survey
    )
)


regional_result = (
    aggregate_by_region(
        region_codes,
        literacy_scores,
        weights,
        threshold,
    )
)


if regional_result.empty:

    st.error(
        "유효한 AI 리터러시 응답과 "
        "지역 코드가 없어 지도를 "
        "만들 수 없습니다."
    )

    st.stop()


boundary_data = get_boundary_table(
    geojson,
    region_level,
)


map_data = boundary_data.merge(
    regional_result,
    on="코드",
    how="left",
)


# 표본이 적은 지역은 색상에서 제외
small_sample_mask = (
    map_data["응답자수"]
    .fillna(0)
    < minimum_region_sample
)


map_data.loc[
    small_sample_mask,
    [
        "AI 리터러시 비율(%)",
        "평균 리터러시 점수",
    ],
] = float("nan")


# =========================================================
# 전국 지표 계산
# =========================================================
valid_mask = (
    literacy_scores.notna()
    & weights.notna()
    & weights.gt(0)
)


valid_weights = weights.where(
    valid_mask
)


valid_weight_sum = (
    valid_weights.sum()
)


if valid_weight_sum > 0:

    national_high_rate = (
        (
            literacy_scores
            .ge(threshold)
            .astype(float)
            * valid_weights
        ).sum()
        / valid_weight_sum
        * 100
    )

    national_average_score = (
        (
            literacy_scores
            * valid_weights
        ).sum()
        / valid_weight_sum
    )

else:

    national_high_rate = (
        float("nan")
    )

    national_average_score = (
        float("nan")
    )


matched_region_count = (
    regional_result[
        "코드"
    ]
    .isin(
        boundary_data["코드"]
    )
    .sum()
)


# =========================================================
# 핵심 지표 표시
# =========================================================
column1, column2, column3, column4 = (
    st.columns(4)
)


with column1:

    st.metric(
        "유효 응답자",
        f"{valid_mask.sum():,}명",
    )


with column2:

    st.metric(
        "전국 고리터러시 비율",
        f"{national_high_rate:.1f}%",
    )


with column3:

    st.metric(
        "전국 평균 점수",
        f"{national_average_score:.2f} / 5",
    )


with column4:

    st.metric(
        "지도 결합 지역",
        f"{matched_region_count:,}개",
    )


st.caption(
    f"원자료: {survey_file_name}"
    f" · 행정단위: {region_level}"
    f" · {weight_note}"
    f" · {code_note}"
)


# =========================================================
# 지도 표시
# =========================================================
figure = create_choropleth(
    map_data=map_data,
    geojson=geojson,
    region_level=region_level,
    threshold=threshold,
)


st.plotly_chart(
    figure,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


if region_level == "시도":

    st.warning(
        "현재 원자료에는 시군구 5자리 코드가 "
        "없어 코드북의 `DM6`를 이용한 "
        "시도 단위 지도를 표시했습니다. "
        "시군구 지도를 만들려면 응답자별 "
        "5자리 `코드` 열이 필요합니다."
    )


# =========================================================
# 매칭되지 않은 코드 확인
# =========================================================
unmatched_codes = regional_result.loc[
    ~regional_result["코드"].isin(
        boundary_data["코드"]
    ),
    "코드",
].tolist()


if unmatched_codes:

    with st.expander(
        "경계 파일과 매칭되지 않은 코드"
    ):

        st.write(
            ", ".join(
                unmatched_codes
            )
        )


# =========================================================
# 지역 순위표
# =========================================================
st.subheader(
    f"{region_level}별 AI 리터러시 순위"
)


ranking = map_data.dropna(
    subset=[
        "AI 리터러시 비율(%)",
    ]
).copy()


ranking = ranking.sort_values(
    "AI 리터러시 비율(%)",
    ascending=False,
)


ranking[
    "AI 리터러시 비율(%)"
] = ranking[
    "AI 리터러시 비율(%)"
].round(1)


ranking[
    "평균 리터러시 점수"
] = ranking[
    "평균 리터러시 점수"
].round(2)


st.dataframe(
    ranking[
        [
            "지역명",
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
# 코드북 문항 확인
# =========================================================
with st.expander(
    "사용한 코드북 Q14 문항"
):

    q14_table = codebook[
        codebook["변수"].isin(
            selected_columns
        )
    ][
        [
            "변수",
            "변수설명",
        ]
    ].drop_duplicates()

    st.dataframe(
        q14_table,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 데이터 점검
# =========================================================
with st.expander(
    "데이터 점검"
):

    st.write(
        f"원자료 크기: "
        f"{survey.shape[0]:,}행 × "
        f"{survey.shape[1]:,}열"
    )

    st.write(
        f"사용한 Q14 문항: "
        f"{len(selected_columns):,}개"
    )

    st.write(
        f"유효한 지역 코드: "
        f"{region_codes.notna().sum():,}개"
    )

    st.write(
        f"유효한 AI 리터러시 응답: "
        f"{valid_mask.sum():,}개"
    )

    st.write(
        f"경계와 결합된 지역: "
        f"{matched_region_count:,}개"
    )
