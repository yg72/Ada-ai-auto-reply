import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.category import Category
from models.context import Context
from models.llm_result import ClassifierResult
from utils.llm import invoke_llm, LLM_USE_CACHE


def classify_conversation(
    context: Context, categories: list[Category], dry_run: bool = False
) -> Category:
    system_prompt = f"""\
You are a helpful assistant that classifies the whole conversation between job seeker and referrer into one of the following categories.
If the conversation is not ended, always mark as "no_reply" if the referrer doesn't reply to the job seeker's latest message.
If multiple categories are applicable, you should choose the one indicating the latest status of the conversation.
You will need to provide confidence score, reason, and referenced message ids (only include the most relevant message ids to the classification).
When classifying, always evaluate the latest messages first.
Never make up facts.

Category Definition:
{categories}
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}
"""

    if context.user_profile:
        user_prompt += f"""
Job Seeker Profile:
{context.user_profile}
"""

    if context.referrer_profile:
        user_prompt += f"""
Referrer Profile:
{context.referrer_profile}
"""

    output_schema = {
        "name": "classifier_result",
        "strict": True,
        "type": "json_schema",
        "schema": ClassifierResult.model_json_schema(),
    }

    if dry_run:
        return system_prompt, user_prompt, output_schema

    response = invoke_llm(
        input=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={"format": output_schema},
        use_cache=LLM_USE_CACHE,
    )

    return ClassifierResult(**json.loads(response.content[0]["text"]))
