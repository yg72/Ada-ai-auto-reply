import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ClassifierResult, InferenceResult, ActionSummarizerResult
from utils.llm import invoke_llm,LLM_USE_CACHE

def inference_results(
    context: Context, 
    classified_category: ClassifierResult, 
    summarized_actions: ActionSummarizerResult,
    dry_run: bool = False
) -> InferenceResult:
    system_prompt = f"""\
You are a helpful assistant that infers the referrer's attitude and possibility to refer the job seeker.
You will be given the conversation history, classified category, summarized actions, job seeker profile (optional), and referrer profile (optional).
You will need to provide inferred results along with confidence score and reason.
When inferring results, always evaluate the latest messages first.
Never make up facts.    

Inference results should be a boolean to indicate if the referral is possible
"""
    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Summarized Actions:
{summarized_actions}
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
        "name": "inference_result",
        "strict": True,
        "type": "json_schema",
        "schema": InferenceResult.model_json_schema(),
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

    return InferenceResult(**json.loads(response.content[0]["text"]))


