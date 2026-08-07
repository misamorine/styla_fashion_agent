import re


def extract_recommendations(llm_response: str):
    """
    Extract clothing recommendations from the LLM response.

    Returns:
        [
            "Structured Tops",
            "Bright Colored Tops",
            ...
        ]
    """

    recommendations = []

    for line in llm_response.split("\n"):

        line = line.strip()

        if line.startswith("-"):

            item = line.lstrip("-").strip()

            # remove "(Priority 5)"
            # remove markdown bold
            item = item.replace("**", "")

# remove (Priority 5)
            item = re.sub(r"\(.*?\)", "", item).strip()

# remove text after colon
            item = item.split(":")[0].strip()

            # remove everything after colon
            item = item.split(":")[0].strip()

            if item:
                recommendations.append(item)

    return recommendations