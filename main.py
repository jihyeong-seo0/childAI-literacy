import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="CSV 데이터 확인",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV 데이터 확인")
st.write("CSV 파일을 업로드하면 `head()`와 `describe()` 결과를 보여줍니다.")

uploaded_file = st.file_uploader(
    "CSV 파일을 선택하세요.",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # 업로드된 파일을 CP949 인코딩으로 읽기
        df = pd.read_csv(uploaded_file, encoding="cp949")

    except UnicodeDecodeError:
        # CP949로 읽지 못하면 UTF-8로 다시 시도
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

    except Exception as error:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {error}")
        st.stop()

    st.success("파일을 성공적으로 읽었습니다.")

    st.subheader("데이터 기본 정보")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("행 개수", f"{df.shape[0]:,}")

    with col2:
        st.metric("열 개수", f"{df.shape[1]:,}")

    st.write("열 이름")
    st.write(list(df.columns))

    st.divider()

    st.subheader("head() 결과")
    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.divider()

    st.subheader("describe() 결과")

    # 숫자형 열뿐 아니라 문자형 열도 함께 요약
    describe_result = df.describe(include="all").transpose()

    st.dataframe(
        describe_result,
        use_container_width=True
    )

    st.divider()

    st.subheader("결측치 개수")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["열 이름", "결측치 개수"]

    st.dataframe(
        missing_values,
        use_container_width=True
    )

else:
    st.info("분석할 CSV 파일을 업로드해주세요.")
