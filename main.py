import json
import re
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
# 파일 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# 보내주신 원자료 이름을 그대로 사용합니다.
DATA_NAME = (
    "(데이터) 청소년의 생성형 AI 이용실태 및 "
    "리터러시 증진방안 연구.csv"
)

DATA_PATH = BASE_DIR / DATA_NAME

# 원자료 ID와 읍면동 코드를 연결하는 추가 파일
MAPPING_NAME = "응답자_읍면동코드.csv"
MAPPING_PATH = BASE_DIR / MAPPING_NAME

# 전국 행정동 경계
GEOJSON_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/"
    "hangjeongdong.geojson"
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

ALL_LITERACY_COLUMNS = [
    column
    for columns in LITERACY_GROUPS.values()
    for column in columns
]


# =========================================================
# CSV 읽기
# =========================================================
def read_csv_flexible(file_path):
    """
    UTF-8-SIG, UTF-8, CP949 순서로 CSV를 읽습니다.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
    ]

    last_error = None

    for encoding in encodings:
        try:
            data = pd.read_csv(
                file_path,
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


@st.cache_data
def load_survey_data(path_string):
    return read_csv_flexible(path_string)


@st.cache_data
def load_mapping_data(path_string):
    return read_csv_flexible(path_string)


# =========================================================
# GeoJSON 읽기
# =========================================================
@st.cache_data(ttl=86400)
def load_geojson(url):
    """
    전국 행정동 GeoJSON을 불러옵니다.
    """

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    with urlopen(
        request,
        timeout=120,
    ) as response:
        text = response.read().decode("utf-8")

    return json.loads(text)


# =========================================================
# 코드 정리
# =========================================================
def normalize_id(value):
    """
    ID를 문자열로 통일합니다.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    text = re.sub(
        r"\.0$",
        "",
        text,
    )

    return text if text else None


def normalize_dong_code(value):
    """
    행정안전부 행정기관코드를 10자리 문자열로 정리합니다.
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

    if len(digits) == 10:
        return digits

    return None


# =========================================================
# 원자료에 읍면동 코드 연결
# =========================================================
def attach_dong_codes(survey):
    """
    원자료 안에 읍면동 코드가 있으면 사용합니다.

    없으면 응답자_읍면동코드.csv를 ID 기준으로 결합합니다.
    """

    code_candidates = [
        "코드",
        "읍면동코드",
        "행정동코드",
        "행정기관코드",
        "지역코드",
        "adm_cd2",
    ]

    # 원자료 자체에서 코드 찾기
    for column in code_candidates:
        if column not in survey.columns:
            continue

        normalized = survey[column].map(
            normalize_dong_code
        )

        if normalized.notna().any():
            result = survey.copy()
            result["읍면동코드"] = normalized

            return (
                result,
                f"원자료의 `{column}` 열 사용",
            )

    # 원자료에 없으면 연결표 사용
    if not MAPPING_PATH.exists():
        return None, None

    mapping = load_mapping_data(
        str(MAPPING_PATH)
    )

    required_columns = {
        "ID",
        "코드",
    }

    if not required_columns.issubset(
        mapping.columns
    ):
        raise ValueError(
            f"{MAPPING_NAME}에는 ID와 코드 열이 필요합니다."
        )

    survey_copy = survey.copy()
    mapping_copy = mapping.copy()

    survey_copy["ID_결합"] = survey_copy[
        "ID"
    ].map(normalize_id)

    mapping_copy["ID_결합"] = mapping_copy[
        "ID"
    ].map(normalize_id)

    mapping_copy["읍면동코드"] = mapping_copy[
        "코드"
    ].map(normalize_dong_code)

    mapping_copy = (
        mapping_copy[
            [
                "ID_결합",
                "읍면동코드",
            ]
        ]
        .dropna(
            subset=[
                "ID_결합",
                "읍면동코드",
            ]
        )
        .drop_duplicates(
            subset=["ID_결합"],
            keep="first",
        )
    )

    result = survey_copy.merge(
        mapping_copy,
        on="ID_결합",
        how="left",
        validate="one_to_one",
    )

    return (
        result,
        f"`{MAPPING_NAME}`을 ID 기준으로 결합",
    )


# =========================================================
# GeoJSON 속성 정리
# =========================================================
def prepare_geojson(geojson):
    """
    GeoJSON의 코드를 10자리 문자열로 통일하고
    지도 결합용 속성을 추가합니다.
    """

    prepared_features = []
    boundary_rows = []

    for feature in geojson.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        raw_code = properties.get(
            "adm_cd2"
        )

        raw_name = properties.get(
            "adm_nm"
        )

        code = normalize_dong_code(
            raw_code
        )

        if code is None:
            continue

        region_name = (
            str(raw_name).strip()
            if raw_name is not None
            else code
        )

        properties[
            "지도코드"
        ] = code

        properties[
            "읍면동명"
        ] = region_name

        feature[
            "properties"
        ] = properties

        prepared_features.append(
            feature
        )

        boundary_rows.append(
            {
                "읍면동코드": code,
                "읍면동": region_name,
            }
        )

    prepared_geojson = {
        "type": "FeatureCollection",
        "features": prepared_features,
    }

    boundary_table = (
        pd.DataFrame(boundary_rows)
        .drop_duplicates(
            subset=["읍면동코드"]
        )
    )

    return (
        prepared_geojson,
        boundary_table,
    )


# =========================================================
# 리터러시 점수 계산
# =========================================================
def calculate_literacy_score(
    data,
    selected_columns,
    minimum_answers,
):
    """
    응답자별 AI 리터러시 평균을 계산합니다.
    """

    answers = (
        data[selected_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    valid_answer_count = (
        answers
        .notna()
        .sum(axis=1)
    )

    average_score = answers.mean(
        axis=1,
        skipna=True,
    )

    return average_score.where(
        valid_answer_count
        >= minimum_answers
    )


def get_weights(data):
    """
    wgt_b가 있으면 가중치를 적용합니다.
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
# 읍면동별 집계
# =========================================================
def aggregate_literacy(
    data,
    literacy_score,
    weights,
    threshold,
):
    """
    읍면동별 고리터러시 비율과 평균 점수를 계산합니다.
    """

    working = pd.DataFrame(
        {
            "읍면동코드": data[
                "읍면동코드"
            ],
            "리터러시점수": literacy_score,
            "가중치": weights,
        }
    )

    working = working.dropna(
        subset=[
            "읍면동코드",
            "리터러시점수",
            "가중치",
        ]
    ).copy()

    working = working[
        working["가중치"] > 0
    ].copy()

    if working.empty:
        return pd.DataFrame()

    working[
        "고리터러시"
    ] = working[
        "리터러시점수"
    ].ge(threshold)

    working[
        "고리터러시가중치"
    ] = (
        working["고리터러시"]
        .astype(float)
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
            "읍면동코드",
            as_index=False,
        )
        .agg(
            응답자수=(
                "리터러시점수",
                "size",
            ),
            가중치합=(
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
        result[
            "고리터러시가중치"
        ]
        / result["가중치합"]
        * 100
    )

    result[
        "평균 리터러시 점수"
    ] = (
        result["점수가중합"]
        / result["가중치합"]
    )

    return result[
        [
            "읍면동코드",
            "응답자수",
            "평균 리터러시 점수",
            "AI 리터러시 비율(%)",
        ]
    ]


# =========================================================
# 전국 수치 계산
# =========================================================
def calculate_national_values(
    scores,
    weights,
    threshold,
):
    valid = (
        scores.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not valid.any():
        return (
            float("nan"),
            float("nan"),
            0,
        )

    valid_scores = scores[valid]
    valid_weights = weights[valid]

    weight_sum = valid_weights.sum()

    average_score = (
        valid_scores
        * valid_weights
    ).sum() / weight_sum

    high_rate = (
        valid_scores
        .ge(threshold)
        .astype(float)
        .mul(valid_weights)
        .sum()
        / weight_sum
        * 100
    )

    return (
        float(average_score),
        float(high_rate),
        int(valid.sum()),
    )


# =========================================================
# 단계구분도
# =========================================================
def create_map(
    map_data,
    geojson,
    threshold,
):
    """
    배경 타일 없이 경계만 표시하는 단계구분도입니다.
    """

    colored_data = map_data.dropna(
        subset=[
            "AI 리터러시 비율(%)",
        ]
    ).copy()

    figure = go.Figure()

    # 데이터가 없는 읍면동의 기본 경계
    figure.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=map_data[
                "읍면동코드"
            ],
            z=[
                0
            ] * len(map_data),
            featureidkey=(
                "properties.지도코드"
            ),
            colorscale=[
                [
                    0,
                    "#eeeeee",
                ],
                [
                    1,
                    "#eeeeee",
                ],
            ],
            showscale=False,
            marker_line_color="#ffffff",
            marker_line_width=0.25,
            hoverinfo="skip",
        )
    )

    # 실제 리터러시 비율
    figure.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=colored_data[
                "읍면동코드"
            ],
            z=colored_data[
                "AI 리터러시 비율(%)"
            ],
            featureidkey=(
                "properties.지도코드"
            ),
            colorscale="Blues",
            zmin=0,
            zmax=100,
            marker_line_color="#ffffff",
            marker_line_width=0.3,
            colorbar={
                "title": {
                    "text": (
                        "AI 리터러시"
                        "<br>비율(%)"
                    ),
                },
                "ticksuffix": "%",
                "thickness": 16,
                "len": 0.7,
            },
            customdata=colored_data[
                [
                    "읍면동",
                    "응답자수",
                    "평균 리터러시 점수",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "AI 리터러시 비율: %{z:.1f}%<br>"
                "평균 점수: %{customdata[2]:.2f} / 5<br>"
                "유효 응답자: %{customdata[1]:,.0f}명"
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
                "전국 청소년 AI 리터러시 단계구분도"
                "<br>"
                f"<sup>읍면동별 평균 {threshold:.1f}점 "
                "이상 응답자 비율</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        height=850,
        margin={
            "l": 0,
            "r": 0,
            "t": 80,
            "b": 0,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return figure


# =========================================================
# 화면
# =========================================================
st.title(
    "🧠 전국 청소년 AI 리터러시 단계구분도"
)

st.caption(
    "읍면동별 AI 리터러시가 높은 청소년의 비율을 표시합니다."
)


# =========================================================
# 원자료 확인
# =========================================================
if not DATA_PATH.exists():
    st.error(
        f"원자료 파일을 찾을 수 없습니다: {DATA_NAME}"
    )

    st.code(
        f"""프로젝트폴더/
├── main.py
├── requirements.txt
├── {DATA_NAME}
└── {MAPPING_NAME}""",
        language="text",
    )

    st.stop()


try:
    survey = load_survey_data(
        str(DATA_PATH)
    )

except Exception as error:
    st.error(
        f"원자료를 읽지 못했습니다: {error}"
    )

    st.stop()


# =========================================================
# 필수 변수 확인
# =========================================================
missing_columns = [
    column
    for column in ALL_LITERACY_COLUMNS
    if column not in survey.columns
]

if missing_columns:
    st.error(
        "원자료에서 다음 Q14 문항을 찾지 못했습니다."
    )

    st.code(
        "\n".join(missing_columns),
        language="text",
    )

    st.stop()


if "ID" not in survey.columns:
    st.error(
        "원자료에 ID 열이 없습니다."
    )

    st.stop()


# =========================================================
# 읍면동 코드 연결
# =========================================================
try:
    survey, code_source = attach_dong_codes(
        survey
    )

except Exception as error:
    st.error(
        f"읍면동 코드를 결합하지 못했습니다: {error}"
    )

    st.stop()


if survey is None:
    st.error(
        "현재 원자료에는 읍면동 코드가 없습니다."
    )

    st.warning(
        f"""
읍면동 지도를 만들려면 `{MAPPING_NAME}` 파일을
GitHub 저장소에 추가해야 합니다.

이 파일에는 각 응답자의 ID와 실제 10자리 행정동 코드가
들어 있어야 합니다.
"""
    )

    st.code(
        """ID,코드
1,1111053000
2,1111054000
3,2611051000""",
        language="text",
    )

    st.info(
        "현재 데이터의 DM6는 17개 시도만 구분하므로 "
        "DM6만으로 읍면동을 알아낼 수 없습니다."
    )

    st.stop()


matched_code_count = (
    survey["읍면동코드"]
    .notna()
    .sum()
)

if matched_code_count == 0:
    st.error(
        "읍면동 코드가 결합된 응답자가 없습니다."
    )

    st.stop()


# =========================================================
# 사이드바 설정
# =========================================================
st.sidebar.header(
    "지도 설정"
)

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
        "AI 리터러시 영역을 하나 이상 선택해주세요."
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
        int(
            len(selected_columns)
            * 0.75
        ),
    ),
)


minimum_sample = st.sidebar.number_input(
    "읍면동 최소 표본 수",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    help=(
        "이 인원보다 응답자가 적은 읍면동은 "
        "지도에서 색칠하지 않습니다."
    ),
)


# =========================================================
# 점수 계산
# =========================================================
literacy_score = calculate_literacy_score(
    survey,
    selected_columns,
    minimum_answers,
)


weights, weight_note = get_weights(
    survey
)


regional_result = aggregate_literacy(
    survey,
    literacy_score,
    weights,
    threshold,
)


if regional_result.empty:
    st.error(
        "유효한 AI 리터러시 응답과 읍면동 코드가 없습니다."
    )

    st.stop()


# =========================================================
# GeoJSON 결합
# =========================================================
try:
    original_geojson = load_geojson(
        GEOJSON_URL
    )

    geojson, boundary_table = prepare_geojson(
        original_geojson
    )

except Exception as error:
    st.error(
        f"전국 행정동 경계 파일을 읽지 못했습니다: {error}"
    )

    st.stop()


map_data = boundary_table.merge(
    regional_result,
    on="읍면동코드",
    how="left",
)


# 최소 표본보다 적은 지역은 값을 숨깁니다.
small_sample = (
    map_data["응답자수"]
    .fillna(0)
    < minimum_sample
)

map_data.loc[
    small_sample,
    [
        "평균 리터러시 점수",
        "AI 리터러시 비율(%)",
    ],
] = float("nan")


# =========================================================
# 전국 수치
# =========================================================
national_average, national_rate, valid_count = (
    calculate_national_values(
        literacy_score,
        weights,
        threshold,
    )
)


survey_codes = set(
    regional_result[
        "읍면동코드"
    ]
)

boundary_codes = set(
    boundary_table[
        "읍면동코드"
    ]
)

matched_regions = len(
    survey_codes
    & boundary_codes
)


# =========================================================
# 주요 수치
# =========================================================
column1, column2, column3, column4 = (
    st.columns(4)
)

with column1:
    st.metric(
        "전체 원자료",
        f"{len(survey):,}명",
    )

with column2:
    st.metric(
        "유효 Q14 응답자",
        f"{valid_count:,}명",
    )

with column3:
    st.metric(
        "전국 평균 점수",
        (
            f"{national_average:.2f} / 5"
            if pd.notna(national_average)
            else "계산 불가"
        ),
    )

with column4:
    st.metric(
        "전국 고리터러시 비율",
        (
            f"{national_rate:.1f}%"
            if pd.notna(national_rate)
            else "계산 불가"
        ),
    )


st.caption(
    f"원자료: {DATA_NAME}"
    f" · 지역 코드: {code_source}"
    f" · {weight_note}"
    f" · 경계와 결합된 읍면동: {matched_regions:,}개"
)


# =========================================================
# 표본 주의
# =========================================================
st.warning(
    """
이 조사는 전국 읍면동별 통계를 산출하기 위한 전수조사가 아닙니다.
읍면동별 응답자가 매우 적을 수 있으므로 결과를 지역의 확정적인
특성으로 해석하면 안 됩니다. 표본 수가 적은 지역은 지도에서
제외하는 것이 안전합니다.
"""
)


# =========================================================
# 지도
# =========================================================
figure = create_map(
    map_data,
    geojson,
    threshold,
)

st.plotly_chart(
    figure,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


# =========================================================
# 순위표
# =========================================================
st.subheader(
    "읍면동별 AI 리터러시 현황"
)

ranking = (
    map_data
    .dropna(
        subset=[
            "AI 리터러시 비율(%)",
        ]
    )
    .sort_values(
        [
            "AI 리터러시 비율(%)",
            "응답자수",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .copy()
)

ranking[
    "평균 리터러시 점수"
] = ranking[
    "평균 리터러시 점수"
].round(2)

ranking[
    "AI 리터러시 비율(%)"
] = ranking[
    "AI 리터러시 비율(%)"
].round(1)


st.dataframe(
    ranking[
        [
            "읍면동",
            "읍면동코드",
            "응답자수",
            "평균 리터러시 점수",
            "AI 리터러시 비율(%)",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 데이터 점검
# =========================================================
with st.expander(
    "데이터 결합 점검"
):
    st.write(
        f"원자료 행 수: {len(survey):,}명"
    )

    st.write(
        f"읍면동 코드가 있는 응답자: "
        f"{survey['읍면동코드'].notna().sum():,}명"
    )

    st.write(
        f"유효한 AI 리터러시 응답자: "
        f"{valid_count:,}명"
    )

    st.write(
        f"경계 파일과 결합된 읍면동: "
        f"{matched_regions:,}개"
    )

    unmatched_codes = sorted(
        survey_codes
        - boundary_codes
    )

    if unmatched_codes:
        st.warning(
            "경계 파일과 연결되지 않은 읍면동 코드가 있습니다."
        )

        st.code(
            "\n".join(
                unmatched_codes[:100]
            ),
            language="text",
        )
