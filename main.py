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

    # 1점 이상, 5점 이하인 응답만 사용
    valid_range = (
        (answers >= 1)
        & (answers <= 5)
    )

    answers = answers.where(
        valid_range
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
        answer_count >= minimum_answers
    )
