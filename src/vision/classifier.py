

def classify_front_shape(
    bust,
    waist,
    hip
):
    """
    BSTI Front Classification

    Returns:
        H
        X
        A
        Y
    """

    # -----------------------------
    # Tolerances
    # -----------------------------

    bust_hip_threshold = 0.06     # 5%
    waist_threshold = 0.15         # 12%

    bust_hip_diff = abs(bust - hip) / max(bust, hip)

    waist_vs_bust = (bust - waist) / bust
    waist_vs_hip = (hip - waist) / hip

    # -----------------------------
    # X Shape
    # Bust ≈ Hip
    # Waist much smaller
    # -----------------------------

    if (
        bust_hip_diff <= bust_hip_threshold
        and
        waist_vs_bust >= waist_threshold
        and
        waist_vs_hip >= waist_threshold
    ):
        return "X"

    # -----------------------------
    # H Shape
    # Everything similar
    # -----------------------------

    if (
        bust_hip_diff <= bust_hip_threshold
        and
        waist_vs_bust < waist_threshold
        and
        waist_vs_hip < waist_threshold
    ):
        return "H"

    # -----------------------------
    # A Shape
    # Hip bigger
    # -----------------------------

    if hip > bust:

        return "A"

    # -----------------------------
    # Y Shape
    # Bust bigger
    # -----------------------------

    return "Y"