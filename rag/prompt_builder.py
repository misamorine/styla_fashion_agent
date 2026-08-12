from langchain_core.prompts import ChatPromptTemplate

BODY_SHAPE_STYLING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Fashion Stylist specializing in 2-view (front & side profile) body shape analysis."),
    (
        "human",
        """The user's body shape profile code is: {body_shape}

Below is trusted styling knowledge retrieved from the fashion knowledge base (covering front silhouette balance and side profile protrusion management).

------------------------
{knowledge}
------------------------

Using ONLY the knowledge above, synthesize a comprehensive, elegant styling guide recommending:
1. Best Tops & Cut
2. Best Bottoms & Fit
3. Best Necklines
4. Best Dresses & Silhouettes
5. Best Jackets & Layering
6. Colors & Visual Balance
7. Prints & Texture Rules
8. Core Styling Tips (Front & Side Profile Harmonies)

For every recommendation explain WHY it suits both the front silhouette and side profile. Keep the answer beautifully formatted.
Do NOT recommend anything that appears in any Avoid section.
Prioritize higher priority items first."""
    ),
])


def build_prompt(body_shape: str, knowledge: str) -> str:
    """Builds styling recommendation prompt using LangChain ChatPromptTemplate."""
    formatted = BODY_SHAPE_STYLING_PROMPT.format_messages(
        body_shape=body_shape,
        knowledge=knowledge
    )
    return formatted[1].content
