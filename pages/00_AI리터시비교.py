from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="집단별 AI 리터러시 비교",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# 데이터 파일 설정
# pages 폴더 안에서도 프로젝트 최상위 폴더를 찾습니다.
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent

if CURRENT_DIR.name == "pages":
    BASE_DIR = CURRENT_DIR.parent
else:
    BASE_DIR = CURRENT_DIR


DATA_NAME = (
    "(데이터) 청소년의 생성형 AI 이용실태 및 "
    "리터러시 증진방안 연구.csv"
)

DATA_PATH = BASE_DIR / DATA_NAME


# =========================================================
# 학년·성별·지역 코드
# =========================================================
GRADE_LABELS = {
    1: "중학교 1학년",
    2: "중학교 2학년",
    3: "중학교 3학년",
    4: "고등학교 1학년",
    5: "고등학교 2학년",
    6: "고등학교 3학년",
}


SEX_LABELS = {
    1: "남자",
    2: "여자",
}


REGION_LABELS = {
    1: "서울",
    2: "부산",
    3: "대구",
    4: "인천",
    5: "광주",
    6: "대전",
    7: "울산",
    8: "세종",
    9: "경기",
    10: "강원",
    11: "충북",
    12: "충남",
    13: "전북",
    14: "전남",
    15: "경북",
    16: "경남",
    17: "제주",
}


# =========================================================
# AI 리터러시 문항
# =========================================================
LITERACY_ITEMS = {
    "정보 사실 확인": "Q14_1_1",
    "정보 신뢰성 판단": "Q14_1_2",
    "정보 편향 판단": "Q14_1_3",
    "질문 방법 이해": "Q14_2_4",
    "전문 용어 활용": "Q14_2_5",
    "AI와 대화 방법": "Q14_2_6",
    "새로운 아이디어 생성": "Q14_3_7",
    "독창적 글쓰기": "Q14_3_8",
    "창의적 결과물 제작": "Q14_3_9",
    "개인정보 보호": "Q14_4_10",
    "저작권 고려": "Q14_4_11",
    "윤리적 사용": "Q14_4_12",
}


LITERACY_DOMAINS = {
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
# AI 교육 필요도
# =========================================================
EDUCATION_NEED_ITEMS = {
    "작동원리 이해": "Q19_1",
    "잘 활용하는 방법": "Q19_2",
    "개인정보·저작권": "Q19_3",
    "오류·편향 확인": "Q19_4",
}


# =========================================================
# AI 교육 경험
# =========================================================
EDUCATION_EXPERIENCE_ITEMS = {
    "작동원리 이해": "Q18_1",
    "잘 활용하는 방법": "Q18_2",
    "개인정보·저작권": "Q18_3",
    "오류·편향 확인": "Q18_4",
}


# =========================================================
# AI 유용성
# =========================================================
USEFULNESS_ITEMS = {
    "효율적으로 공부": "Q12_1",
    "원하는 정보 획득": "Q12_2",
    "학습에 도움": "Q12_3",
    "학습 성과 향상": "Q12_4",
    "생산적으로 공부": "Q12_5",
}


# =========================================================
# AI 도움 분야
# =========================================================
HELP_ITEMS = {
    "코딩": "Q13_1",
    "필요한 자료 찾기": "Q13_2",
    "자료 요약": "Q13_3",
    "번역": "Q13_4",
    "자기소개서 작성": "Q13_5",
    "논문·보고서 작성": "Q13_6",
    "수학·과학 문제풀이": "Q13_7",
    "시·소설 등 창작": "Q13_8",
}


# =========================================================
# 만족도
# =========================================================
SATISFACTION_ITEMS = {
    "사용 편리성": "Q15_1",
    "답변 유용성": "Q15_2",
    "답변 흥미성": "Q15_3",
    "답변 신뢰성": "Q15_4",
}


# =========================================================
# 향후 사용 의향
# =========================================================
INTENTION_ITEMS = {
    "향후 적극 활용": "Q16_1",
    "친구에게 권유": "Q16_2",
}


# =========================================================
# 부정적 현상 인식
# =========================================================
RISK_ITEMS = {
    "출처 없이 과제 제출": "Q17_1",
    "저작권 침해": "Q17_2",
    "허위정보 확산": "Q17_3",
    "인간 창의성 저하": "Q17_4",
}


# =========================================================
# 데이터 읽기
# =========================================================
@st.cache_data
def load_data(path_string):
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
# 숫자형 문항 정리
# =========================================================
def numeric_answers(
    data,
    column,
    maximum,
):
    """
    문항을 숫자로 변환하고 정상 범위만 남깁니다.
    """

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    return values.where(
        values.between(
            1,
            maximum,
        )
    )


# =========================================================
# 가중치 가져오기
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
            return weights

    return pd.Series(
        1.0,
        index=data.index,
    )


# =========================================================
# 가중평균
# =========================================================
def weighted_mean(
    data,
    values,
):
    """
    가중평균과 유효 응답자 수를 계산합니다.
    """

    weights = get_weights(data)

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = (
        values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not valid.any():
        return float("nan"), 0

    result = (
        values.loc[valid]
        .mul(
            weights.loc[valid]
        )
        .sum()
        / weights.loc[valid].sum()
    )

    return (
        float(result),
        int(valid.sum()),
    )


# =========================================================
# 가중 비율
# =========================================================
def weighted_rate(
    data,
    condition,
    valid_values,
):
    """
    가중 비율을 백분율로 계산합니다.
    """

    weights = get_weights(data)

    valid = (
        valid_values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not valid.any():
        return float("nan"), 0

    rate = (
        condition.loc[valid]
        .astype(float)
        .mul(
            weights.loc[valid]
        )
        .sum()
        / weights.loc[valid].sum()
        * 100
    )

    return (
        float(rate),
        int(valid.sum()),
    )


# =========================================================
# 응답자별 종합 점수
# =========================================================
def row_mean(
    data,
    columns,
    maximum,
    minimum_answers,
):
    """
    여러 문항의 응답자별 평균을 계산합니다.
    """

    frame = (
        data[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    frame = frame.where(
        (frame >= 1)
        & (frame <= maximum)
    )

    valid_count = (
        frame
        .notna()
        .sum(axis=1)
    )

    scores = frame.mean(
        axis=1,
        skipna=True,
    )

    return scores.where(
        valid_count
        >= minimum_answers
    )


def composite_score(
    data,
    columns,
    maximum,
    minimum_answers,
):
    """
    여러 문항의 종합 가중평균을 계산합니다.
    """

    scores = row_mean(
        data,
        columns,
        maximum,
        minimum_answers,
    )

    return weighted_mean(
        data,
        scores,
    )


# =========================================================
# 문항별 비교 자료 생성
# =========================================================
def build_item_comparison(
    selected_data,
    reference_data,
    items,
    maximum,
    selected_name,
    reference_name,
):
    rows = []

    for label, column in items.items():

        selected_score, selected_n = (
            weighted_mean(
                selected_data,
                numeric_answers(
                    selected_data,
                    column,
                    maximum,
                ),
            )
        )

        reference_score, reference_n = (
            weighted_mean(
                reference_data,
                numeric_answers(
                    reference_data,
                    column,
                    maximum,
                ),
            )
        )

        rows.extend(
            [
                {
                    "지표": label,
                    "집단": selected_name,
                    "점수": selected_score,
                    "유효 응답자": selected_n,
                },
                {
                    "지표": label,
                    "집단": reference_name,
                    "점수": reference_score,
                    "유효 응답자": reference_n,
                },
            ]
        )

    return pd.DataFrame(rows)


# =========================================================
# 리터러시 영역별 비교 자료
# =========================================================
def build_domain_comparison(
    selected_data,
    reference_data,
    selected_name,
    reference_name,
):
    rows = []

    for label, columns in LITERACY_DOMAINS.items():

        selected_score, selected_n = (
            composite_score(
                selected_data,
                columns,
                maximum=5,
                minimum_answers=2,
            )
        )

        reference_score, reference_n = (
            composite_score(
                reference_data,
                columns,
                maximum=5,
                minimum_answers=2,
            )
        )

        rows.extend(
            [
                {
                    "지표": label,
                    "집단": selected_name,
                    "점수": selected_score,
                    "유효 응답자": selected_n,
                },
                {
                    "지표": label,
                    "집단": reference_name,
                    "점수": reference_score,
                    "유효 응답자": reference_n,
                },
            ]
        )

    return pd.DataFrame(rows)


# =========================================================
# 막대그래프
# =========================================================
def make_bar_chart(
    data,
    title,
    maximum,
):
    chart_data = data.dropna(
        subset=["점수"]
    ).copy()

    if chart_data.empty:
        return None

    figure = px.bar(
        chart_data,
        x="지표",
        y="점수",
        color="집단",
        barmode="group",
        text="점수",
        hover_data={
            "유효 응답자": True,
            "점수": ":.2f",
        },
        title=title,
    )

    figure.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside",
        cliponaxis=False,
    )

    figure.update_yaxes(
        range=[
            0,
            maximum + 0.35,
        ],
        title=(
            f"평균 점수 "
            f"({maximum}점 만점)"
        ),
    )

    figure.update_xaxes(
        title=None
    )

    figure.update_layout(
        height=520,
        legend_title_text="",
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
    )

    return figure


# =========================================================
# 차이 표
# =========================================================
def show_difference_table(
    data,
    selected_name,
    reference_name,
):
    pivot = (
        data
        .pivot_table(
            index="지표",
            columns="집단",
            values="점수",
            aggfunc="first",
        )
        .reset_index()
    )

    if selected_name not in pivot.columns:
        pivot[selected_name] = float("nan")

    if reference_name not in pivot.columns:
        pivot[reference_name] = float("nan")

    pivot["차이"] = (
        pivot[selected_name]
        - pivot[reference_name]
    )

    display = pivot[
        [
            "지표",
            selected_name,
            reference_name,
            "차이",
        ]
    ].copy()

    display[selected_name] = (
        display[selected_name]
        .round(2)
    )

    display[reference_name] = (
        display[reference_name]
        .round(2)
    )

    display["차이"] = (
        display["차이"]
        .round(2)
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    valid_difference = display.dropna(
        subset=["차이"]
    )

    if not valid_difference.empty:

        largest = valid_difference.loc[
            valid_difference[
                "차이"
            ].abs().idxmax()
        ]

        if largest["차이"] > 0:
            direction = "높습니다"
        elif largest["차이"] < 0:
            direction = "낮습니다"
        else:
            direction = "같습니다"

        st.info(
            f"가장 큰 차이는 **{largest['지표']}**입니다. "
            f"선택 집단이 비교 집단보다 "
            f"**{abs(largest['차이']):.2f}점 {direction}**."
        )


# =========================================================
# 그래프와 차이표 표시
# =========================================================
def show_comparison_section(
    data,
    title,
    maximum,
    selected_name,
    reference_name,
):
    figure = make_bar_chart(
        data,
        title,
        maximum,
    )

    if figure is None:
        st.warning(
            "이 조건에서는 계산 가능한 응답이 없습니다."
        )
        return

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    show_difference_table(
        data,
        selected_name,
        reference_name,
    )


# =========================================================
# 화면 제목
# =========================================================
st.title(
    "📊 연령·성별·지역별 AI 인식 비교"
)

st.caption(
    "선택한 학생 집단과 전국 또는 나머지 응답자의 "
    "AI 리터러시, 교육 필요도, 교육 경험, "
    "유용성 등을 비교합니다."
)


st.info(
    "원자료에는 실제 만 나이 변수가 없습니다. "
    "따라서 이 페이지에서는 `DQ1`의 학교 학년을 "
    "나이 구분 기준으로 사용합니다."
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
├── pages/
│   └── 2_집단별_비교.py
└── {DATA_NAME}""",
        language="text",
    )

    st.stop()


try:
    survey = load_data(
        str(DATA_PATH)
    )

except Exception as error:

    st.error(
        "데이터 파일을 읽는 중 오류가 발생했습니다."
    )

    st.exception(error)

    st.stop()


# =========================================================
# 필수 변수 확인
# =========================================================
required_columns = {
    "DQ1",
    "SQ3",
    "DM6",
    "Q4",
    "wgt_b",
}

required_columns.update(
    LITERACY_ITEMS.values()
)

required_columns.update(
    EDUCATION_NEED_ITEMS.values()
)

required_columns.update(
    EDUCATION_EXPERIENCE_ITEMS.values()
)

required_columns.update(
    USEFULNESS_ITEMS.values()
)

required_columns.update(
    HELP_ITEMS.values()
)

required_columns.update(
    SATISFACTION_ITEMS.values()
)

required_columns.update(
    INTENTION_ITEMS.values()
)

required_columns.update(
    RISK_ITEMS.values()
)


missing_columns = sorted(
    required_columns
    - set(survey.columns)
)


if missing_columns:

    st.error(
        "분석에 필요한 변수가 데이터에 없습니다."
    )

    st.code(
        "\n".join(
            missing_columns
        ),
        language="text",
    )

    st.stop()


# =========================================================
# 필터용 숫자 변수
# =========================================================
survey["DQ1_num"] = pd.to_numeric(
    survey["DQ1"],
    errors="coerce",
).astype("Int64")


survey["SQ3_num"] = pd.to_numeric(
    survey["SQ3"],
    errors="coerce",
).astype("Int64")


survey["DM6_num"] = pd.to_numeric(
    survey["DM6"],
    errors="coerce",
).astype("Int64")


# =========================================================
# 집단 선택
# =========================================================
st.subheader(
    "비교할 집단 선택"
)

filter_col1, filter_col2, filter_col3, filter_col4 = (
    st.columns(
        [
            1,
            1,
            1.2,
            1.5,
        ]
    )
)


with filter_col1:

    selected_grade_label = st.selectbox(
        "나이·학년",
        options=[
            "전체",
            *GRADE_LABELS.values(),
        ],
    )


with filter_col2:

    selected_sex_label = st.selectbox(
        "성별",
        options=[
            "전체",
            *SEX_LABELS.values(),
        ],
    )


with filter_col3:

    selected_region_label = st.selectbox(
        "지역",
        options=[
            "전체",
            *REGION_LABELS.values(),
        ],
    )


with filter_col4:

    comparison_mode = st.radio(
        "비교 기준",
        options=[
            "전국 전체",
            "선택 조건을 제외한 나머지",
        ],
    )


# =========================================================
# 선택 조건 적용
# =========================================================
selected_mask = pd.Series(
    True,
    index=survey.index,
)

selected_parts = []


if selected_grade_label != "전체":

    selected_grade_code = next(
        code
        for code, label
        in GRADE_LABELS.items()
        if label == selected_grade_label
    )

    selected_mask &= (
        survey["DQ1_num"]
        .eq(selected_grade_code)
        .fillna(False)
    )

    selected_parts.append(
        selected_grade_label
    )


if selected_sex_label != "전체":

    selected_sex_code = next(
        code
        for code, label
        in SEX_LABELS.items()
        if label == selected_sex_label
    )

    selected_mask &= (
        survey["SQ3_num"]
        .eq(selected_sex_code)
        .fillna(False)
    )

    selected_parts.append(
        selected_sex_label
    )


if selected_region_label != "전체":

    selected_region_code = next(
        code
        for code, label
        in REGION_LABELS.items()
        if label == selected_region_label
    )

    selected_mask &= (
        survey["DM6_num"]
        .eq(selected_region_code)
        .fillna(False)
    )

    selected_parts.append(
        selected_region_label
    )


selected_data = survey.loc[
    selected_mask
].copy()


if selected_parts:
    selected_full_label = " · ".join(
        selected_parts
    )
else:
    selected_full_label = "전체 응답자"


selected_chart_name = "선택 집단"


if comparison_mode == "전국 전체":

    reference_data = survey.copy()

    reference_full_label = (
        "전국 전체"
    )

    reference_chart_name = (
        "전국 전체"
    )

else:

    reference_data = survey.loc[
        ~selected_mask
    ].copy()

    reference_full_label = (
        "선택 조건을 제외한 나머지"
    )

    reference_chart_name = (
        "나머지 응답자"
    )


if selected_data.empty:

    st.error(
        "선택한 조건에 해당하는 응답자가 없습니다."
    )

    st.stop()


if reference_data.empty:

    st.warning(
        "비교 집단이 비어 있습니다. "
        "비교 기준을 `전국 전체`로 바꾸거나 "
        "학년·성별·지역 중 하나 이상을 선택하세요."
    )

    st.stop()


st.markdown(
    f"**선택 집단:** {selected_full_label}  \n"
    f"**비교 집단:** {reference_full_label}"
)


# =========================================================
# 핵심 지표 계산
# =========================================================

# AI 이용률
selected_q4 = pd.to_numeric(
    selected_data["Q4"],
    errors="coerce",
)

selected_use_rate, selected_use_n = (
    weighted_rate(
        selected_data,
        selected_q4.eq(2),
        selected_q4,
    )
)


reference_q4 = pd.to_numeric(
    reference_data["Q4"],
    errors="coerce",
)

reference_use_rate, reference_use_n = (
    weighted_rate(
        reference_data,
        reference_q4.eq(2),
        reference_q4,
    )
)


# 종합 AI 리터러시
all_literacy_columns = list(
    LITERACY_ITEMS.values()
)


selected_literacy, selected_literacy_n = (
    composite_score(
        selected_data,
        all_literacy_columns,
        maximum=5,
        minimum_answers=9,
    )
)


reference_literacy, reference_literacy_n = (
    composite_score(
        reference_data,
        all_literacy_columns,
        maximum=5,
        minimum_answers=9,
    )
)


# 교육 필요도
selected_need, selected_need_n = (
    composite_score(
        selected_data,
        list(
            EDUCATION_NEED_ITEMS.values()
        ),
        maximum=4,
        minimum_answers=3,
    )
)


reference_need, reference_need_n = (
    composite_score(
        reference_data,
        list(
            EDUCATION_NEED_ITEMS.values()
        ),
        maximum=4,
        minimum_answers=3,
    )
)


# 교육 경험
selected_experience, selected_experience_n = (
    composite_score(
        selected_data,
        list(
            EDUCATION_EXPERIENCE_ITEMS.values()
        ),
        maximum=4,
        minimum_answers=3,
    )
)


reference_experience, reference_experience_n = (
    composite_score(
        reference_data,
        list(
            EDUCATION_EXPERIENCE_ITEMS.values()
        ),
        maximum=4,
        minimum_answers=3,
    )
)


# =========================================================
# 핵심 지표 표시
# =========================================================
metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


with metric1:

    st.metric(
        "선택 집단 응답자",
        f"{len(selected_data):,}명",
        delta=(
            f"비교 집단 "
            f"{len(reference_data):,}명"
        ),
        delta_color="off",
    )


with metric2:

    st.metric(
        "AI 이용률",
        f"{selected_use_rate:.1f}%",
        delta=(
            f"{selected_use_rate - reference_use_rate:+.1f}%p"
        ),
    )


with metric3:

    st.metric(
        "종합 AI 리터러시",
        f"{selected_literacy:.2f} / 5",
        delta=(
            f"{selected_literacy - reference_literacy:+.2f}점"
        ),
    )


with metric4:

    st.metric(
        "AI 교육 필요도",
        f"{selected_need:.2f} / 4",
        delta=(
            f"{selected_need - reference_need:+.2f}점"
        ),
    )


if len(selected_data) < 30:

    st.warning(
        "선택 집단의 응답자가 30명보다 적습니다. "
        "평균 차이가 표본 변동에 민감할 수 있으므로 "
        "해석에 주의하세요."
    )


st.caption(
    "막대그래프의 차이는 `wgt_b` 가중평균 기준입니다. "
    "AI 리터러시·유용성·만족도 문항은 "
    "실제로 해당 문항에 응답한 학생만 포함됩니다."
)


# =========================================================
# 탭
# =========================================================
(
    tab_summary,
    tab_literacy,
    tab_education,
    tab_usefulness,
    tab_help,
    tab_attitude,
    tab_risk,
) = st.tabs(
    [
        "핵심 비교",
        "AI 리터러시",
        "교육 필요·경험",
        "AI 유용성",
        "도움 분야",
        "만족·사용의향",
        "위험 인식",
    ]
)


# =========================================================
# 핵심 비교
# =========================================================
with tab_summary:

    summary_data = pd.DataFrame(
        {
            "지표": [
                "AI 이용률(%)",
                "종합 AI 리터러시(5점)",
                "교육 필요도(4점)",
                "교육 경험(4점)",
            ],
            selected_chart_name: [
                selected_use_rate,
                selected_literacy,
                selected_need,
                selected_experience,
            ],
            reference_chart_name: [
                reference_use_rate,
                reference_literacy,
                reference_need,
                reference_experience,
            ],
        }
    )

    summary_data["차이"] = (
        summary_data[selected_chart_name]
        - summary_data[reference_chart_name]
    )

    st.dataframe(
        summary_data.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "핵심 지표마다 점수 범위가 다르기 때문에 "
        "이 탭에서는 표로 제시합니다. "
        "다른 탭의 막대그래프는 같은 척도의 문항끼리 비교합니다."
    )


# =========================================================
# AI 리터러시
# =========================================================
with tab_literacy:

    domain_data = build_domain_comparison(
        selected_data,
        reference_data,
        selected_chart_name,
        reference_chart_name,
    )

    show_comparison_section(
        domain_data,
        "AI 리터러시 4개 영역 비교",
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    with st.expander(
        "12개 세부 문항 보기"
    ):

        literacy_item_data = (
            build_item_comparison(
                selected_data,
                reference_data,
                LITERACY_ITEMS,
                maximum=5,
                selected_name=selected_chart_name,
                reference_name=reference_chart_name,
            )
        )

        show_comparison_section(
            literacy_item_data,
            "AI 리터러시 세부 문항 비교",
            maximum=5,
            selected_name=selected_chart_name,
            reference_name=reference_chart_name,
        )


# =========================================================
# 교육 필요도·경험
# =========================================================
with tab_education:

    st.subheader(
        "AI 교육 필요도"
    )

    need_data = build_item_comparison(
        selected_data,
        reference_data,
        EDUCATION_NEED_ITEMS,
        maximum=4,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        need_data,
        "생성형 AI 교육 필요도 비교",
        maximum=4,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    st.caption(
        "점수가 높을수록 해당 교육이 "
        "더 필요하다고 인식합니다."
    )

    st.divider()

    st.subheader(
        "AI 교육 경험"
    )

    experience_data = build_item_comparison(
        selected_data,
        reference_data,
        EDUCATION_EXPERIENCE_ITEMS,
        maximum=4,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        experience_data,
        "생성형 AI 교육 경험 비교",
        maximum=4,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    st.divider()

    st.subheader(
        "교육 필요도와 경험의 격차"
    )

    st.caption(
        "값이 클수록 필요하다고 느끼는 정도에 비해 "
        "실제 교육 경험이 부족하다는 뜻입니다."
    )

    gap_rows = []

    for label in EDUCATION_NEED_ITEMS:

        need_column = (
            EDUCATION_NEED_ITEMS[label]
        )

        experience_column = (
            EDUCATION_EXPERIENCE_ITEMS[label]
        )

        selected_need_values = numeric_answers(
            selected_data,
            need_column,
            4,
        )

        selected_experience_values = numeric_answers(
            selected_data,
            experience_column,
            4,
        )

        selected_gap, selected_gap_n = (
            weighted_mean(
                selected_data,
                (
                    selected_need_values
                    - selected_experience_values
                ),
            )
        )

        reference_need_values = numeric_answers(
            reference_data,
            need_column,
            4,
        )

        reference_experience_values = numeric_answers(
            reference_data,
            experience_column,
            4,
        )

        reference_gap, reference_gap_n = (
            weighted_mean(
                reference_data,
                (
                    reference_need_values
                    - reference_experience_values
                ),
            )
        )

        gap_rows.extend(
            [
                {
                    "지표": label,
                    "집단": selected_chart_name,
                    "점수": selected_gap,
                    "유효 응답자": selected_gap_n,
                },
                {
                    "지표": label,
                    "집단": reference_chart_name,
                    "점수": reference_gap,
                    "유효 응답자": reference_gap_n,
                },
            ]
        )

    gap_data = pd.DataFrame(
        gap_rows
    )

    gap_chart = px.bar(
        gap_data,
        x="지표",
        y="점수",
        color="집단",
        barmode="group",
        text="점수",
        hover_data={
            "유효 응답자": True,
            "점수": ":.2f",
        },
        title="교육 필요도 - 교육 경험 격차",
    )

    gap_chart.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside",
        cliponaxis=False,
    )

    gap_chart.update_yaxes(
        title="필요도 - 교육 경험",
        rangemode="tozero",
    )

    gap_chart.update_xaxes(
        title=None
    )

    gap_chart.update_layout(
        height=520,
        legend_title_text="",
    )

    st.plotly_chart(
        gap_chart,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    show_difference_table(
        gap_data,
        selected_chart_name,
        reference_chart_name,
    )


# =========================================================
# AI 유용성
# =========================================================
with tab_usefulness:

    usefulness_data = build_item_comparison(
        selected_data,
        reference_data,
        USEFULNESS_ITEMS,
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        usefulness_data,
        "생성형 AI 유용성 인식 비교",
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )


# =========================================================
# AI 도움 분야
# =========================================================
with tab_help:

    help_data = build_item_comparison(
        selected_data,
        reference_data,
        HELP_ITEMS,
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        help_data,
        "생성형 AI가 도움이 되는 분야 비교",
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )


# =========================================================
# 만족도와 사용의향
# =========================================================
with tab_attitude:

    st.subheader(
        "생성형 AI 만족도"
    )

    satisfaction_data = build_item_comparison(
        selected_data,
        reference_data,
        SATISFACTION_ITEMS,
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        satisfaction_data,
        "생성형 AI 만족도 비교",
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    st.divider()

    st.subheader(
        "향후 사용의향"
    )

    intention_data = build_item_comparison(
        selected_data,
        reference_data,
        INTENTION_ITEMS,
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        intention_data,
        "향후 생성형 AI 사용의향 비교",
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )


# =========================================================
# 위험 인식
# =========================================================
with tab_risk:

    risk_data = build_item_comparison(
        selected_data,
        reference_data,
        RISK_ITEMS,
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    show_comparison_section(
        risk_data,
        "생성형 AI의 부정적 현상에 대한 심각성 인식",
        maximum=5,
        selected_name=selected_chart_name,
        reference_name=reference_chart_name,
    )

    st.caption(
        "점수가 높을수록 해당 문제를 "
        "더 심각하게 인식한다는 뜻입니다."
    )


# =========================================================
# 분석 주의사항
# =========================================================
with st.expander(
    "분석 기준과 주의사항"
):

    st.markdown(
        """
- 모든 평균과 비율은 `wgt_b`가 유효한 경우 표준화 가중치를 적용합니다.
- 막대그래프의 차이는 인과관계가 아니라 집단 간 평균 차이입니다.
- AI 리터러시, 유용성, 도움 정도, 만족도 문항은 생성형 AI 이용 경험자에게만 제시됐을 가능성이 있어 유효 응답자 수가 전체 응답자 수보다 적습니다.
- 선택 집단의 표본 수가 작으면 평균이 불안정할 수 있습니다.
- `전국 전체`는 선택 집단을 포함한 전체 응답자입니다.
- 선택 집단과 완전히 분리된 비교를 원하면 `선택 조건을 제외한 나머지`를 선택하세요.
        """
    )
