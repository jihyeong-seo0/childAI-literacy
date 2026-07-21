import json
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
# 데이터 파일 설정
# 보내주신 파일 이름을 그대로 사용합니다.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_NAME = (
    "(데이터) 청소년의 생성형 AI 이용실태 및 "
    "리터러시 증진방안 연구.csv"
)

DATA_PATH = BASE_DIR / DATA_NAME


# 전국 시도 경계 GeoJSON
GEOJSON_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/"
    "main/data/boundaries/sido_kr.geojson"
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
# DM6 코드 → 행정구역 2자리 코드
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
# CSV 읽기
# =========================================================
def read_csv_flexible(file_path):
    """
    UTF-8-SIG, CP949, UTF-8 순서로 CSV를 읽습니다.
    """

    last_error = None

    for encoding in (
        "utf-8-sig",
        "cp949",
        "utf-8",
    ):
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
def load_survey(path_string):
    return read_csv_flexible(
        Path(path_string)
    )


# =========================================================
# GeoJSON 읽기
# =========================================================
@st.cache_data(ttl=86400)
def load_geojson(url):
    """
    GitHub에서 전국 시도 경계를 읽습니다.
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
        text = response.read().decode(
            "utf-8"
        )

    return json.loads(text)


# =========================================================
# 시도 코드 정리
# =========================================================
def normalize_sido_code(value):
    """
    시도 코드를 2자리 문자열로 변환합니다.
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    if not digits:
        return None

    return digits.zfill(2)[-2:]


# =========================================================
# GeoJSON 속성 정리
# =========================================================
def prepare_geojson(geojson):
    """
    GeoJSON의 코드와 시도 이름을 정리합니다.
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

        code = normalize_sido_code(
            properties.get("코드")
        )

        if code is None:
            continue

        region_name = None

        name_candidates = [
            "시도",
            "sidonm",
            "CTP_KOR_NM",
            "name",
        ]

        for candidate in name_candidates:
            candidate_value = properties.get(
                candidate
            )

            if candidate_value not in (
                None,
                "",
            ):
                region_name = str(
                    candidate_value
                ).strip()

                break

        if region_name is None:
            region_name = code

        properties["코드"] = code
        properties["시도"] = region_name

        feature["properties"] = properties

        prepared_features.append(
            feature
        )

        boundary_rows.append(
            {
                "코드": code,
                "시도": region_name,
            }
        )

    prepared_geojson = {
        "type": "FeatureCollection",
        "features": prepared_features,
    }

    boundary_table = (
        pd.DataFrame(
            boundary_rows
        )
        .drop_duplicates(
            subset="코드"
        )
    )

    return (
        prepared_geojson,
        boundary_table,
    )


# =========================================================
# 가중치 설정
# =========================================================
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
                "표준화 가중치 wgt_b 적용",
            )

    default_weights = pd.Series(
        1.0,
        index=data.index,
    )

    return (
        default_weights,
        "가중치 없음",
    )


# =========================================================
# 개인별 AI 리터러시 계산
# =========================================================
def calculate_literacy_scores(
    data,
    columns,
    minimum_answers,
):
    """
    선택한 Q14 문항의 개인별 평균을 계산합니다.
    """

    answers = (
        data[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # 정상적인 1~5점 응답만 사용
    answers = answers.where(
        answers.between(
            1,
            5,
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
        >= minimum_answers
    )


# =========================================================
# 시도별 집계
# =========================================================
def aggregate_by_sido(
    region_codes,
    literacy_scores,
    weights,
    threshold,
):
    """
    시도별 AI 리터러시 비율과 평균 점수를 계산합니다.
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
    ).copy()

    working = working[
        working["가중치"] > 0
    ].copy()

    if working.empty:
        return pd.DataFrame()

    working[
        "고리터러시가중치"
    ] = (
        working["리터러시점수"]
        .ge(threshold)
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
            "코드",
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
        result["고리터러시가중치"]
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
            "코드",
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
    """
    전국 평균과 고리터러시 비율을 계산합니다.
    """

    valid = (
        scores.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not valid.any():
        return (
            0,
            float("nan"),
            float("nan"),
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
        int(valid.sum()),
        float(average_score),
        float(high_rate),
    )


# =========================================================
# 단계구분도 생성
# =========================================================
def create_choropleth(
    map_data,
    geojson,
    threshold,
):
    """
    배경 타일 없이 시도 경계만 표시합니다.
    """

    colored_data = map_data.dropna(
        subset=[
            "AI 리터러시 비율(%)",
        ]
    ).copy()

    figure = go.Figure()

    # 데이터가 없는 지역의 기본 회색 경계
    figure.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=map_data["코드"],
            z=[0] * len(map_data),
            featureidkey="properties.코드",
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
            marker_line_color="white",
            marker_line_width=0.8,
            hoverinfo="skip",
        )
    )

    # AI 리터러시 비율 색상
    figure.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=colored_data[
                "코드"
            ],
            z=colored_data[
                "AI 리터러시 비율(%)"
            ],
            featureidkey="properties.코드",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            marker_line_color="white",
            marker_line_width=0.9,
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
                    "시도",
                    "응답자수",
                    "평균 리터러시 점수",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "AI 리터러시 비율: "
                "%{z:.1f}%<br>"
                "평균 리터러시 점수: "
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
                f"<sup>시도별 개인 평균 "
                f"{threshold:.1f}점 이상 "
                "응답자 비율</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        height=780,
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
# 화면 제목
# =========================================================
st.title(
    "🧠 전국 청소년 AI 리터러시 단계구분도"
)

st.caption(
    "시도별로 AI 리터러시가 높은 "
    "청소년의 비율을 표시합니다."
)


with st.expander(
    "계산 기준"
):
    st.markdown(
        """
- `Q14`의 12개 AI 리터러시 문항을 응답자별로 평균 냅니다.
- 기본 기준은 개인 평균 **4.0점 이상**입니다.
- 지역별 비율은 유효한 Q14 응답자 중 기준 점수 이상인 응답자의 가중 비율입니다.
- `wgt_b`가 있으면 표준화 가중치를 적용합니다.
- Q14 문항에 답하지 않은 응답자는 리터러시 비율의 분모에서 제외됩니다.
        """
    )


# =========================================================
# 데이터 파일 확인
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
    survey = load_survey(
        str(DATA_PATH)
    )

except Exception as error:
    st.error(
        f"데이터 파일을 읽지 못했습니다: {error}"
    )

    st.stop()


# =========================================================
# 필수 열 확인
# =========================================================
all_q14_columns = [
    column
    for group_columns in LITERACY_GROUPS.values()
    for column in group_columns
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
    st.error(
        "데이터에서 필요한 열을 찾지 못했습니다."
    )

    st.code(
        "\n".join(
            missing_columns
        ),
        language="text",
    )

    st.stop()


# =========================================================
# 사이드바 옵션
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
    max_value=len(
        selected_columns
    ),
    value=max(
        1,
        round(
            len(selected_columns)
            * 0.75
        ),
    ),
)


minimum_sample = st.sidebar.number_input(
    "시도 최소 표본 수",
    min_value=1,
    max_value=1000,
    value=30,
    step=1,
    help=(
        "이보다 유효 응답자가 적은 시도는 "
        "지도에서 색칠하지 않습니다."
    ),
)


# =========================================================
# DM6 → 시도 코드 변환
# =========================================================
dm6 = pd.to_numeric(
    survey["DM6"],
    errors="coerce",
).astype("Int64")


dm6_to_code = {
    dm6_value: region[0]
    for dm6_value, region
    in DM6_TO_REGION.items()
}


dm6_to_name = {
    dm6_value: region[1]
    for dm6_value, region
    in DM6_TO_REGION.items()
}


survey["코드"] = dm6.map(
    dm6_to_code
)

survey["시도"] = dm6.map(
    dm6_to_name
)


# =========================================================
# AI 리터러시 계산
# =========================================================
literacy_scores = (
    calculate_literacy_scores(
        survey,
        selected_columns,
        minimum_answers,
    )
)


weights, weight_note = get_weights(
    survey
)


regional_result = aggregate_by_sido(
    survey["코드"],
    literacy_scores,
    weights,
    threshold,
)


if regional_result.empty:
    st.error(
        "유효한 지역 코드와 "
        "AI 리터러시 응답이 없습니다."
    )

    st.stop()


# =========================================================
# 시도 경계 읽기
# =========================================================
try:
    raw_geojson = load_geojson(
        GEOJSON_URL
    )

    geojson, boundary_table = (
        prepare_geojson(
            raw_geojson
        )
    )

except Exception as error:
    st.error(
        f"시도 경계 파일을 읽지 못했습니다: {error}"
    )

    st.stop()


# 코드북의 시도 명칭을 우선 사용
sido_name_table = pd.DataFrame(
    [
        {
            "코드": code,
            "설문 시도": name,
        }
        for code, name
        in DM6_TO_REGION.values()
    ]
)


map_data = (
    boundary_table
    .merge(
        sido_name_table,
        on="코드",
        how="left",
    )
    .merge(
        regional_result,
        on="코드",
        how="left",
    )
)


map_data["시도"] = (
    map_data["설문 시도"]
    .fillna(
        map_data["시도"]
    )
)


map_data = map_data.drop(
    columns=[
        "설문 시도",
    ]
)


# 최소 표본 미달 지역은 지도 색상 제외
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
# 전국 수치 계산
# =========================================================
(
    valid_count,
    national_average,
    national_high_rate,
) = calculate_national_values(
    literacy_scores,
    weights,
    threshold,
)


# =========================================================
# 핵심 지표
# =========================================================
metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


with metric1:
    st.metric(
        "전체 응답자",
        f"{len(survey):,}명",
    )


with metric2:
    st.metric(
        "유효 Q14 응답자",
        f"{valid_count:,}명",
    )


with metric3:
    st.metric(
        "전국 평균 점수",
        (
            f"{national_average:.2f} / 5"
            if pd.notna(
                national_average
            )
            else "계산 불가"
        ),
    )


with metric4:
    st.metric(
        "전국 고리터러시 비율",
        (
            f"{national_high_rate:.1f}%"
            if pd.notna(
                national_high_rate
            )
            else "계산 불가"
        ),
    )


st.caption(
    f"데이터: {DATA_NAME}"
    f" · 지역 변수: DM6"
    f" · {weight_note}"
)


# =========================================================
# 지도
# =========================================================
figure = create_choropleth(
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
# 시도별 순위표
# =========================================================
st.subheader(
    "시도별 AI 리터러시 현황"
)


ranking = (
    map_data
    .dropna(
        subset=[
            "AI 리터러시 비율(%)",
        ]
    )
    .sort_values(
        "AI 리터러시 비율(%)",
        ascending=False,
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
        f"{len(selected_columns)}개"
    )

    st.write(
        f"유효 Q14 응답자: "
        f"{valid_count:,}명"
    )

    st.write(
        "지역 코드가 연결된 응답자: "
        f"{survey['코드'].notna().sum():,}명"
    )

    unknown_dm6 = sorted(
        dm6[
            ~dm6.isin(
                DM6_TO_REGION
            )
        ]
        .dropna()
        .unique()
    )

    if unknown_dm6:
        st.warning(
            "알 수 없는 DM6 코드가 있습니다."
        )

        st.code(
            ", ".join(
                map(
                    str,
                    unknown_dm6,
                )
            ),
            language="text",
        )
