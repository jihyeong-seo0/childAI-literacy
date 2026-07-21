from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="청소년 생성형 AI 분석",
    page_icon="📊",
    layout="wide",
)

CODEBOOK_NAME = "(코드북) 청소년의 생성형 AI 이용실태 및 리터러시 증진방안 연구.csv"
CODEBOOK_PATH = Path(__file__).parent / CODEBOOK_NAME


# 필터에 사용할 변수
FILTERS = {
    "학교급": "SQ0_3",
    "학년": "DQ1",
    "성별": "SQ3",
    "지역": "DM6",
}


# AI 리터러시 영역
LITERACY = {
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


# 생성형 AI 서비스
SERVICES = {
    "챗GPT": "Q6_1",
    "소라": "Q6_2",
    "빙": "Q6_3",
    "제미나이": "Q6_4",
    "미드저니": "Q6_5",
    "달리": "Q6_6",
    "스냅챗 마이 AI": "Q6_7",
    "클로버엑스": "Q6_8",
    "뤼튼": "Q6_9",
}


# 교육 경험과 교육 필요성
EDUCATION = {
    "작동원리": (
        "Q18_1",
        "Q19_1",
    ),
    "활용 방법": (
        "Q18_2",
        "Q19_2",
    ),
    "개인정보·저작권": (
        "Q18_3",
        "Q19_3",
    ),
    "오류·편향 확인": (
        "Q18_4",
        "Q19_4",
    ),
}


# Q7 사용시간 범주의 대표 시간
# 2시간 이상은 135분으로 가정
TIME_MINUTES = {
    1: 15,
    2: 45,
    3: 75,
    4: 105,
    5: 135,
}


# =========================================================
# CSV 읽기
# =========================================================
def read_csv(source):
    """
    CP949, UTF-8-SIG, UTF-8 순서로 CSV 파일을 읽습니다.
    """

    encodings = [
        "cp949",
        "utf-8-sig",
        "utf-8",
    ]

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

    raise ValueError("CSV 파일의 인코딩을 확인할 수 없습니다.")


@st.cache_data
def load_codebook(path):
    """
    코드북을 불러오고 빈 변수명을 앞의 값으로 채웁니다.
    """

    codebook = read_csv(path)

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

    if not required_columns.issubset(codebook.columns):
        raise ValueError(
            "코드북에 변수, 변수설명, 변수값, 레이블 열이 필요합니다."
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
def load_raw_data(raw_bytes):
    """
    업로드된 학생별 원자료를 불러옵니다.
    """

    data = read_csv(
        BytesIO(raw_bytes)
    )

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    return data


# =========================================================
# 계산 함수
# =========================================================
def number(data, column):
    """
    특정 열을 숫자형으로 변환합니다.
    """

    if column not in data.columns:
        return pd.Series(
            index=data.index,
            dtype="float64",
        )

    return pd.to_numeric(
        data[column],
        errors="coerce",
    )


def weights_for(data):
    """
    표준화 가중치 wgt_b가 있으면 사용합니다.
    """

    if "wgt_b" in data.columns:
        weights = number(
            data,
            "wgt_b",
        )

        weights = weights.where(
            weights > 0
        )

        if weights.notna().any():
            return (
                weights,
                "표준화 가중치(wgt_b) 적용",
            )

    return (
        pd.Series(
            1.0,
            index=data.index,
        ),
        "가중치 없음",
    )


def weighted_mean(values, weights):
    """
    가중평균을 계산합니다.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return float("nan")

    weighted_sum = (
        values[valid]
        * weights[valid]
    ).sum()

    weight_sum = weights[valid].sum()

    if weight_sum == 0:
        return float("nan")

    return float(
        weighted_sum
        / weight_sum
    )


def weighted_rate(
    condition,
    values,
    weights,
):
    """
    가중 비율을 백분율로 계산합니다.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return float("nan")

    weighted_count = (
        condition[valid].astype(float)
        * weights[valid]
    ).sum()

    weight_sum = weights[valid].sum()

    if weight_sum == 0:
        return float("nan")

    return float(
        weighted_count
        / weight_sum
        * 100
    )


def row_average(
    data,
    columns,
):
    """
    여러 문항의 응답자별 평균을 계산합니다.
    """

    existing_columns = [
        column
        for column in columns
        if column in data.columns
    ]

    if not existing_columns:
        return pd.Series(
            index=data.index,
            dtype="float64",
        )

    numeric_data = (
        data[existing_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    return numeric_data.mean(
        axis=1,
        skipna=True,
    )


def code_labels(
    codebook,
    variable,
):
    """
    코드북에서 숫자 코드와 레이블을 연결합니다.
    """

    table = codebook[
        codebook["변수"]
        .astype(str)
        .eq(variable)
    ].copy()

    table["변수값"] = pd.to_numeric(
        table["변수값"],
        errors="coerce",
    )

    table = table.dropna(
        subset=[
            "변수값",
            "레이블",
        ]
    )

    return dict(
        zip(
            table["변수값"].astype(float),
            table["레이블"].astype(str),
        )
    )


def show_value(
    value,
    digits=1,
    suffix="",
):
    """
    계산 결과를 화면 표시용 문자열로 변환합니다.
    """

    if pd.isna(value):
        return "계산 불가"

    return (
        f"{value:,.{digits}f}"
        f"{suffix}"
    )


# =========================================================
# 사이드바 필터
# =========================================================
def apply_filters(
    data,
    codebook,
):
    result = data.copy()

    st.sidebar.header(
        "분석 대상 필터"
    )

    st.sidebar.caption(
        "선택한 조건은 모든 결과에 적용됩니다."
    )

    for title, variable in FILTERS.items():

        if variable not in result.columns:
            continue

        label_map = code_labels(
            codebook,
            variable,
        )

        codes = sorted(
            number(
                result,
                variable,
            )
            .dropna()
            .unique()
        )

        display_to_code = {
            label_map.get(
                float(code),
                f"코드 {code:g}",
            ): float(code)
            for code in codes
        }

        selected = st.sidebar.multiselect(
            f"{title} 선택",
            options=list(
                display_to_code.keys()
            ),
            default=list(
                display_to_code.keys()
            ),
            key=f"filter_{variable}",
        )

        if not selected:
            return result.iloc[0:0]

        selected_codes = [
            display_to_code[item]
            for item in selected
        ]

        result = result[
            number(
                result,
                variable,
            ).isin(selected_codes)
        ]

    return result


# =========================================================
# 화면 제목
# =========================================================
st.title(
    "📊 1. 전체 현황"
)

st.write(
    "생성형 AI 이용률, 리터러시, 사용시간, "
    "서비스 이용과 교육 격차를 한 화면에서 확인합니다."
)


# =========================================================
# 코드북 확인
# =========================================================
if not CODEBOOK_PATH.exists():

    st.error(
        f"코드북 파일이 없습니다: {CODEBOOK_NAME}"
    )

    st.write(
        "main.py와 코드북 CSV 파일을 같은 폴더에 넣어주세요."
    )

    st.code(
        """프로젝트폴더/
├── main.py
├── requirements.txt
└── (코드북) 청소년의 생성형 AI 이용실태 및 리터러시 증진방안 연구.csv""",
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
# 학생별 설문 원자료 업로드
# =========================================================
st.sidebar.header(
    "설문 원자료"
)

uploaded_file = st.sidebar.file_uploader(
    "학생별 원자료 CSV 업로드",
    type=["csv"],
    help=(
        "한 행이 학생 한 명이고 "
        "Q4, Q14_1_1, DM6 같은 열이 있는 파일입니다."
    ),
)


# 업로드된 파일을 세션에 저장
# 나중에 다른 페이지에서도 같은 데이터를 사용할 수 있습니다.
if uploaded_file is not None:

    st.session_state[
        "survey_raw_bytes"
    ] = uploaded_file.getvalue()

    st.session_state[
        "survey_file_name"
    ] = uploaded_file.name


# 원자료가 아직 없는 경우
if "survey_raw_bytes" not in st.session_state:

    st.info(
        "코드북은 정상적으로 연결됐습니다. "
        "왼쪽에서 학생별 설문 원자료 CSV를 업로드하면 "
        "실제 결과가 계산됩니다."
    )

    st.warning(
        "코드북만으로는 이용률과 평균 점수를 계산할 수 없습니다."
    )

    preview = pd.DataFrame(
        {
            "표시할 지표": [
                "응답자 수",
                "생성형 AI 이용률",
                "종합 AI 리터러시",
                "추정 평균 사용시간",
                "가장 많이 쓰는 서비스",
                "가장 큰 교육 격차",
            ],
            "계산 기준": [
                "원자료 행 수",
                "Q4=2의 비율",
                "Q14 12개 문항 평균",
                "Q7 범주를 분으로 환산",
                "Q6에서 3점 이상 비율",
                "Q19 필요성 - Q18 경험",
            ],
        }
    )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


# =========================================================
# 원자료 불러오기
# =========================================================
try:
    raw = load_raw_data(
        st.session_state[
            "survey_raw_bytes"
        ]
    )

except Exception as error:

    st.error(
        f"원자료를 읽지 못했습니다: {error}"
    )

    st.stop()


# 필터 적용
filtered = apply_filters(
    raw,
    codebook,
)

if filtered.empty:

    st.warning(
        "선택한 필터에 해당하는 응답자가 없습니다."
    )

    st.stop()


# 가중치 설정
weights, weight_note = weights_for(
    filtered
)

st.caption(
    f"원자료: "
    f"{st.session_state.get('survey_file_name', '업로드 파일')}"
    f" · 코드북: {CODEBOOK_NAME}"
    f" · {weight_note}"
)


# =========================================================
# 핵심 지표 계산
# =========================================================

# 생성형 AI 이용률
# Q4: 1=없음, 2=있음
q4 = number(
    filtered,
    "Q4",
)

ai_use_rate = weighted_rate(
    q4.eq(2),
    q4,
    weights,
)


# 종합 AI 리터러시
all_literacy_columns = [
    item
    for columns in LITERACY.values()
    for item in columns
]

overall_literacy = weighted_mean(
    row_average(
        filtered,
        all_literacy_columns,
    ),
    weights,
)


# 평균 사용시간
q7 = number(
    filtered,
    "Q7",
)

use_minutes = q7.map(
    TIME_MINUTES
)

average_minutes = weighted_mean(
    use_minutes,
    weights,
)


# =========================================================
# 서비스별 이용률
# =========================================================
service_rows = []

for service, variable in SERVICES.items():

    if variable not in filtered.columns:
        continue

    values = number(
        filtered,
        variable,
    )

    service_rows.append(
        {
            "서비스": service,
            "이용자 비율(%)": weighted_rate(
                values.ge(3),
                values,
                weights,
            ),
            "평균 빈도(5점)": weighted_mean(
                values,
                weights,
            ),
        }
    )


if service_rows:

    service_df = pd.DataFrame(
        service_rows
    )

    service_df = service_df.sort_values(
        "이용자 비율(%)",
        ascending=False,
    ).reset_index(
        drop=True
    )

else:

    service_df = pd.DataFrame()


top_service = "계산 불가"
top_service_rate = float("nan")

if not service_df.empty:

    top_service = service_df.iloc[
        0
    ]["서비스"]

    top_service_rate = service_df.iloc[
        0
    ]["이용자 비율(%)"]


# =========================================================
# 교육 격차 계산
# =========================================================
education_rows = []

for area, (
    experience_column,
    need_column,
) in EDUCATION.items():

    if (
        experience_column not in filtered.columns
        or need_column not in filtered.columns
    ):
        continue

    experience = number(
        filtered,
        experience_column,
    )

    need = number(
        filtered,
        need_column,
    )

    gap = need - experience

    education_rows.append(
        {
            "교육 영역": area,
            "교육 경험(4점)": weighted_mean(
                experience,
                weights,
            ),
            "교육 필요성(4점)": weighted_mean(
                need,
                weights,
            ),
            "교육 격차": weighted_mean(
                gap,
                weights,
            ),
        }
    )


if education_rows:

    education_df = pd.DataFrame(
        education_rows
    )

    education_df = education_df.sort_values(
        "교육 격차",
        ascending=False,
    ).reset_index(
        drop=True
    )

else:

    education_df = pd.DataFrame()


largest_gap_area = "계산 불가"
largest_gap = float("nan")

if not education_df.empty:

    largest_gap_area = education_df.iloc[
        0
    ]["교육 영역"]

    largest_gap = education_df.iloc[
        0
    ]["교육 격차"]


# =========================================================
# 핵심 지표 화면
# =========================================================
st.subheader(
    "핵심 지표"
)

column1, column2, column3 = st.columns(
    3
)

with column1:

    st.metric(
        "분석 응답자",
        f"{len(filtered):,}명",
    )

with column2:

    st.metric(
        "생성형 AI 이용률",
        show_value(
            ai_use_rate,
            1,
            "%",
        ),
    )

with column3:

    st.metric(
        "종합 AI 리터러시",
        show_value(
            overall_literacy,
            2,
            " / 5",
        ),
    )


column4, column5, column6 = st.columns(
    3
)

with column4:

    st.metric(
        "추정 평균 사용시간",
        show_value(
            average_minutes,
            1,
            "분",
        ),
        help=(
            "Q7 응답 범주를 "
            "15·45·75·105·135분으로 환산한 추정치입니다."
        ),
    )

with column5:

    st.metric(
        "가장 많이 이용하는 서비스",
        top_service,
    )

    if not pd.isna(
        top_service_rate
    ):

        st.caption(
            f"가끔 이상 이용자 {top_service_rate:.1f}%"
        )

with column6:

    st.metric(
        "가장 큰 교육 격차",
        largest_gap_area,
        help=(
            "교육 필요성 Q19에서 "
            "교육 경험 Q18을 뺀 값입니다."
        ),
    )

    if not pd.isna(
        largest_gap
    ):

        st.caption(
            f"필요성 - 경험 = {largest_gap:+.2f}점"
        )


st.divider()


# =========================================================
# AI 리터러시 영역별 평균
# =========================================================
st.subheader(
    "AI 리터러시 영역별 평균"
)

literacy_rows = []

for area, columns in LITERACY.items():

    score = weighted_mean(
        row_average(
            filtered,
            columns,
        ),
        weights,
    )

    if not pd.isna(score):

        literacy_rows.append(
            {
                "리터러시 영역": area,
                "평균 점수": score,
            }
        )


literacy_df = pd.DataFrame(
    literacy_rows
)


if literacy_df.empty:

    st.warning(
        "Q14 문항이 없어 리터러시를 계산할 수 없습니다."
    )

else:

    literacy_chart = literacy_df.set_index(
        "리터러시 영역"
    )[
        [
            "평균 점수",
        ]
    ]

    st.bar_chart(
        literacy_chart
    )

    literacy_table = literacy_df.copy()

    literacy_table[
        "평균 점수"
    ] = literacy_table[
        "평균 점수"
    ].round(2)

    st.dataframe(
        literacy_table,
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# =========================================================
# 서비스별 이용 현황
# =========================================================
st.subheader(
    "서비스별 이용 현황"
)

st.caption(
    "Q6에서 '가끔 사용한다(3점)' 이상으로 응답한 비율입니다."
)


if service_df.empty:

    st.warning(
        "Q6 문항이 없어 서비스별 이용률을 계산할 수 없습니다."
    )

else:

    service_chart = service_df.set_index(
        "서비스"
    )[
        [
            "이용자 비율(%)",
        ]
    ]

    st.bar_chart(
        service_chart
    )

    service_table = service_df.copy()

    service_table[
        "이용자 비율(%)"
    ] = service_table[
        "이용자 비율(%)"
    ].round(1)

    service_table[
        "평균 빈도(5점)"
    ] = service_table[
        "평균 빈도(5점)"
    ].round(2)

    st.dataframe(
        service_table,
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# =========================================================
# 교육 경험과 필요성
# =========================================================
st.subheader(
    "교육 경험과 교육 필요성"
)

st.caption(
    "교육 격차가 클수록 필요성에 비해 실제 교육 경험이 부족합니다."
)


if education_df.empty:

    st.warning(
        "Q18·Q19 문항이 없어 교육 격차를 계산할 수 없습니다."
    )

else:

    education_chart = education_df.set_index(
        "교육 영역"
    )[
        [
            "교육 경험(4점)",
            "교육 필요성(4점)",
        ]
    ]

    st.bar_chart(
        education_chart
    )

    education_table = education_df.copy()

    for column in [
        "교육 경험(4점)",
        "교육 필요성(4점)",
        "교육 격차",
    ]:

        education_table[
            column
        ] = education_table[
            column
        ].round(2)

    st.dataframe(
        education_table,
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# =========================================================
# 결과 자동 요약
# =========================================================
st.subheader(
    "결과 자동 요약"
)


if not pd.isna(
    ai_use_rate
):

    st.markdown(
        f"- 생성형 AI 이용률은 "
        f"**{ai_use_rate:.1f}%**입니다."
    )


if not literacy_df.empty:

    strongest = literacy_df.loc[
        literacy_df[
            "평균 점수"
        ].idxmax()
    ]

    weakest = literacy_df.loc[
        literacy_df[
            "평균 점수"
        ].idxmin()
    ]

    st.markdown(
        f"- 가장 높은 리터러시는 "
        f"**{strongest['리터러시 영역']} "
        f"({strongest['평균 점수']:.2f}점)**이고, "
        f"가장 낮은 영역은 "
        f"**{weakest['리터러시 영역']} "
        f"({weakest['평균 점수']:.2f}점)**입니다."
    )


if not service_df.empty:

    st.markdown(
        f"- 가장 널리 이용되는 서비스는 "
        f"**{top_service}**이며, "
        f"가끔 이상 이용자는 "
        f"**{top_service_rate:.1f}%**입니다."
    )


if not education_df.empty:

    st.markdown(
        f"- 교육 격차가 가장 큰 영역은 "
        f"**{largest_gap_area} "
        f"({largest_gap:+.2f}점)**입니다."
    )


# =========================================================
# 데이터 점검
# =========================================================
with st.expander(
    "데이터 점검"
):

    st.write(
        f"전체 원자료: "
        f"**{raw.shape[0]:,}행 × {raw.shape[1]:,}열**"
    )

    st.write(
        f"필터 적용 후: "
        f"**{filtered.shape[0]:,}명**"
    )

    st.write(
        f"가중치: "
        f"**{weight_note}**"
    )

    required_variables = (
        [
            "Q4",
            "Q7",
            "wgt_b",
        ]
        + all_literacy_columns
        + list(
            SERVICES.values()
        )
        + [
            variable
            for pair in EDUCATION.values()
            for variable in pair
        ]
        + list(
            FILTERS.values()
        )
    )

    missing_variables = sorted(
        {
            variable
            for variable in required_variables
            if variable not in raw.columns
        }
    )

    if missing_variables:

        st.warning(
            "원자료에서 찾지 못한 변수: "
            + ", ".join(
                missing_variables
            )
        )

    else:

        st.success(
            "이 페이지에 필요한 주요 변수가 모두 있습니다."
        )
