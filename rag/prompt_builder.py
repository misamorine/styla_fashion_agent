from langchain_core.prompts import ChatPromptTemplate

BODY_SHAPE_STYLING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Fashion Stylist."),
    (
        "human",
        """The user's detected body shape is: {body_shape}

Below is trusted styling knowledge retrieved from the fashion knowledge base.

------------------------
{knowledge}
------------------------

Using ONLY the knowledge above, recommend:
1. Best Tops
2. Best Bottoms
3. Best Necklines
4. Best Dresses
5. Best Jackets
6. Colors
7. Prints
8. Styling Tips

For every recommendation explain WHY it suits this body shape. Keep the answer well formatted.
Do not recommend anything that appears in the Avoid section.
If multiple recommendations exist, prioritize higher priority items first."""
    ),
])


def build_prompt(body_shape: str, knowledge: str) -> str:
    """Builds styling recommendation prompt using LangChain ChatPromptTemplate."""
    formatted = BODY_SHAPE_STYLING_PROMPT.format_messages(
        body_shape=body_shape,
        knowledge=knowledge
    )
    return formatted[1].content