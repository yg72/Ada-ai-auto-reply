import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ClassifierResult, ActionSummarizerResult
from utils.llm import invoke_llm,LLM_USE_CACHE

def summarize_actions(
    context: Context, classified_category: ClassifierResult, dry_run: bool = False
) -> ActionSummarizerResult:

    system_prompt = f"""\
You are a helpful assistant that summarizes actions for the job seeker to take based on the conversation history and classified category.
You will be given the conversation history, classified category, job seeker profile (optional), and referrer profile (optional).
You will need to provide a summary of actions along with confidence score and reason.
When summarizing actions, always evaluate the latest messages first.
Never make up facts.

Actions Examples (but not limited to):
1. Provide the latest resume
2. Zoom call availability
3. Job IDs
5. Job Website links
6. LinkedIn profile link
7. Email
8. Phone number
"""    

    user_prompt = f"""\
Conversation Messages:
{context.messages}      

Classified Category:
{classified_category}
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
        "name": "action_summarizer_result",
        "strict": True,
        "type": "json_schema",
        "schema": ActionSummarizerResult.model_json_schema(),
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

    return ActionSummarizerResult(**json.loads(response.content[0]["text"]))
