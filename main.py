from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="청소년 생성형 AI 연구 코드북",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 파일 설정
# 보내주신 파일명을 그대로 사용합니다.
# --------------------------------------------------
FILE_NAME = "(코드북) 청소년의 생성형 AI 이용실태 및 리터러시 증진방안 연구.csv"
FILE_PATH = Path(__file__).parent / FILE_NAME


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data(file_path: Path):
    """
    CP949 인코딩으로 CSV 파일을 불러옵니다.
    CP949로 읽지 못하면 UTF-8-SIG로 다시 시도합니다.
    """
    try:
        data = pd.read_csv(
            file_path,
            encoding="cp949",
            na_values=["#N/A"],
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            na_values=["#N/A"],
        )

    return data


# --------------------------------------------------
# 특정 변수의 코드표 만들기
# --------------------------------------------------
def get_code_table(data: pd.DataFrame, variable_name: str):
    """
    변수명을 기준으로 변수값과 레이블을 추출합니다.
    """
    result = data.loc[
        data["변수"].eq(variable_name),
        ["변수", "변수설명", "변수값", "레이블"],
    ].copy()

    result["변수값"] = pd.to_numeric(
        result["변수값"],
        errors="coerce",
    ).astype("Int64")

    return result.reset_index(drop=True)


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("📊 청소년의 생성형 AI 이용실태 연구 코드북")

st.caption(f"사용 파일: {FILE_NAME}")


# --------------------------------------------------
# 파일 존재 여부 확인
# --------------------------------------------------
if not FILE_PATH.exists():
    st.error("CSV 파일을 찾을 수 없습니다.")

    st.warning(
        "main.py와 CSV 파일을 같은 GitHub 폴더에 넣어주세요."
    )

    st.code(
        """
프로젝트폴더/
├── main.py
├── requirements.txt
└── (코드북) 청소년의 생성형 AI 이용실태 및 리터러시 증진방안 연구.csv
        """,
        language="text",
    )

    st.stop()


# --------------------------------------------------
# 데이터 읽기
# --------------------------------------------------
try:
    df_raw = load_data(FILE_PATH)

except Exception as error:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()


# 코드북은 첫 번째 행에만 변수명과 변수설명이 있으므로
# 아래 빈칸에 앞의 값을 채워 넣습니다.
df = df_raw.copy()

df[["변수", "변수설명"]] = df[
    ["변수", "변수설명"]
].ffill()


# --------------------------------------------------
# 기본 정보
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="전체 행",
        value=f"{df_raw.shape[0]:,}개",
    )

with col2:
    st.metric(
        label="전체 열",
        value=f"{df_raw.shape[1]:,}개",
    )

with col3:
    variable_count = df["변수"].nunique()

    st.metric(
        label="전체 변수",
        value=f"{variable_count:,}개",
    )


st.info(
    """
이 파일은 응답자별 설문 원자료가 아니라 변수와 코드값을 설명하는 코드북입니다.

따라서 연령별·지역별 실제 응답자 수가 아니라,
각 숫자 코드가 어떤 학교급, 학년, 지역을 뜻하는지 보여줍니다.
"""
)


# --------------------------------------------------
# 코드표 생성
# --------------------------------------------------
school_table = get_code_table(df, "SQ0_3")
grade_table = get_code_table(df, "DQ1")
region_table = get_code_table(df, "DM6")


# --------------------------------------------------
# 탭 구성
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏫 연령대·학교급",
        "🎓 나이·학년",
        "🗺️ 지역",
        "🔎 변수 검색",
        "📋 head() / describe()",
    ]
)


# --------------------------------------------------
# 연령대 및 학교급
# --------------------------------------------------
with tab1:
    st.header("연령대·학교급 구분")

    st.write(
        """
코드북에는 정확한 만 나이 변수가 없습니다.
대신 `SQ0_3` 학교급 변수를 이용해 학교급별로 구분할 수 있습니다.
"""
    )

    selected_school = st.selectbox(
        label="확인할 학교급을 선택하세요.",
        options=["전체"] + school_table["레이블"].dropna().tolist(),
        key="school_select",
    )

    if selected_school == "전체":
        displayed_school = school_table
    else:
        displayed_school = school_table[
            school_table["레이블"].eq(selected_school)
        ]

    st.dataframe(
        displayed_school,
        use_container_width=True,
        hide_index=True,
    )

    if selected_school != "전체":
        selected_row = displayed_school.iloc[0]

        st.success(
            f"""
선택한 학교급은 **{selected_row["레이블"]}**이며,
데이터에서 사용하는 코드값은 **{selected_row["변수값"]}**입니다.
"""
        )


# --------------------------------------------------
# 나이 및 학년
# --------------------------------------------------
with tab2:
    st.header("나이·학년 구분")

    st.write(
        """
코드북에는 실제 만 나이가 포함되어 있지 않습니다.
`DQ1` 학교급 및 학년 변수를 이용해 학년별로 구분합니다.
"""
    )

    selected_grade = st.selectbox(
        label="확인할 학년을 선택하세요.",
        options=["전체"] + grade_table["레이블"].dropna().tolist(),
        key="grade_select",
    )

    if selected_grade == "전체":
        displayed_grade = grade_table
    else:
        displayed_grade = grade_table[
            grade_table["레이블"].eq(selected_grade)
        ]

    st.dataframe(
        displayed_grade,
        use_container_width=True,
        hide_index=True,
    )

    if selected_grade != "전체":
        selected_row = displayed_grade.iloc[0]

        st.success(
            f"""
선택한 학년은 **{selected_row["레이블"]}**이며,
데이터에서 사용하는 코드값은 **{selected_row["변수값"]}**입니다.
"""
        )


# --------------------------------------------------
# 지역
# --------------------------------------------------
with tab3:
    st.header("지역별 구분")

    st.write(
        """
`DM6` 시도 변수를 기준으로 전국 17개 시·도를 확인할 수 있습니다.
"""
    )

    selected_region = st.selectbox(
        label="확인할 지역을 선택하세요.",
        options=["전체"] + region_table["레이블"].dropna().tolist(),
        key="region_select",
    )

    if selected_region == "전체":
        displayed_region = region_table
    else:
        displayed_region = region_table[
            region_table["레이블"].eq(selected_region)
        ]

    st.dataframe(
        displayed_region,
        use_container_width=True,
        hide_index=True,
    )

    if selected_region != "전체":
        selected_row = displayed_region.iloc[0]

        st.success(
            f"""
선택한 지역은 **{selected_row["레이블"]}**이며,
데이터에서 사용하는 코드값은 **{selected_row["변수값"]}**입니다.
"""
        )


# --------------------------------------------------
# 전체 변수 검색
# --------------------------------------------------
with tab4:
    st.header("전체 변수 검색")

    st.write(
        "변수명이나 변수 설명을 입력해 코드북 내용을 검색할 수 있습니다."
    )

    search_word = st.text_input(
        label="검색어",
        placeholder="예: 생성형 AI, 학교급, 성별, 지역",
    )

    if search_word:
        search_condition = (
            df["변수"]
            .astype(str)
            .str.contains(
                search_word,
                case=False,
                na=False,
            )
            |
            df["변수설명"]
            .astype(str)
            .str.contains(
                search_word,
                case=False,
                na=False,
            )
            |
            df["레이블"]
            .astype(str)
            .str.contains(
                search_word,
                case=False,
                na=False,
            )
        )

        search_result = df.loc[
            search_condition,
            ["변수", "변수설명", "변수값", "레이블"],
        ].copy()

        st.write(f"검색 결과: {len(search_result):,}개")

        st.dataframe(
            search_result,
            use_container_width=True,
            hide_index=True,
        )

    else:
        variable_summary = (
            df[["변수", "변수설명"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        st.write(f"전체 변수: {len(variable_summary):,}개")

        st.dataframe(
            variable_summary,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# head()와 describe()
# --------------------------------------------------
with tab5:
    st.header("원본 데이터 확인")

    st.subheader("head() 결과")

    head_count = st.slider(
        label="표시할 행 개수",
        min_value=5,
        max_value=30,
        value=5,
        step=5,
    )

    st.dataframe(
        df_raw.head(head_count),
        use_container_width=True,
    )

    st.divider()

    st.subheader("describe() 결과")

    describe_result = (
        df_raw
        .describe(include="all")
        .transpose()
    )

    st.dataframe(
        describe_result,
        use_container_width=True,
    )

    st.divider()

    st.subheader("열별 결측치")

    missing_result = pd.DataFrame(
        {
            "열 이름": df_raw.columns,
            "결측치 개수": df_raw.isna().sum().values,
            "결측치 비율(%)": (
                df_raw.isna().mean().values * 100
            ).round(2),
        }
    )

    st.dataframe(
        missing_result,
        use_container_width=True,
        hide_index=True,
    )
