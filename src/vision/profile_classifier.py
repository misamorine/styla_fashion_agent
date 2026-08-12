"""
Side-profile body shape classifier.
Classifies side-view body protrusions (chest/bust, belly, butt) relative to torso height.

Side Profile Types:
    I  = normal / no prominent protrusion
    P  = prominent chest
    b  = prominent belly
    B  = prominent chest + belly
    S  = prominent chest + butt
    d  = prominent butt
    db = prominent belly + butt
    dB = prominent chest + belly + butt
"""

def classify_side_shape(
    bust: float,
    belly: float,
    butt: float,
    torso_height: float
) -> str:
    """
    Side-profile body classification based on bust, belly, and butt protrusion.

    Args:
        bust: Protrusion distance of bust in pixels.
        belly: Protrusion distance of belly in pixels.
        butt: Protrusion distance of butt in pixels.
        torso_height: Distance between shoulder center and hip center in pixels.

    Returns:
        Side profile classification code: 'I', 'P', 'b', 'B', 'S', 'd', 'db', or 'dB'.
    """
    if torso_height <= 0:
        return "I"

    # --------------------------------------------------------
    # NORMALIZE BY TORSO HEIGHT
    # --------------------------------------------------------
    bust_ratio = bust / torso_height
    belly_ratio = belly / torso_height
    butt_ratio = butt / torso_height

    # --------------------------------------------------------
    # TUNABLE THRESHOLDS
    # --------------------------------------------------------
    MIN_PROMINENCE = 0.18
    RELATIVE_THRESHOLD = 0.85


    values = {
        "bust": bust_ratio,
        "belly": belly_ratio,
        "butt": butt_ratio
    }

    max_value = max(values.values()) if values else 0.0

    # --------------------------------------------------------
    # CHECK PROMINENCE
    # --------------------------------------------------------
    bust_prominent = (
        bust_ratio >= MIN_PROMINENCE
        and bust_ratio >= max_value * RELATIVE_THRESHOLD
    )

    belly_prominent = (
        belly_ratio >= MIN_PROMINENCE
        and belly_ratio >= max_value * RELATIVE_THRESHOLD
    )

    butt_prominent = (
        butt_ratio >= MIN_PROMINENCE
        and butt_ratio >= max_value * RELATIVE_THRESHOLD
    )

    # --------------------------------------------------------
    # CLASSIFICATION LOGIC
    # --------------------------------------------------------
    # 1. EVERYTHING PROMINENT
    if bust_prominent and belly_prominent and butt_prominent:
        return "dB"

    # 2. CHEST + BELLY
    if bust_prominent and belly_prominent:
        return "B"

    # 3. CHEST + BUTT
    if bust_prominent and butt_prominent:
        return "S"

    # 4. BELLY + BUTT
    if belly_prominent and butt_prominent:
        return "db"

    # 5. ONLY CHEST
    if bust_prominent:
        return "P"

    # 6. ONLY BELLY
    if belly_prominent:
        return "b"

    # 7. ONLY BUTT
    if butt_prominent:
        return "d"

    # 8. NOTHING PROMINENT
    return "I"
