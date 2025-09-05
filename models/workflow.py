from models.base import BaseModel
from models.context import Context
from models.llm_result import (
    ClassifierResult,
    MessageGeneratorResult,
    TopicSuggesterResult,
    ActionSummarizerResult,
    InferenceResult
)


class State(BaseModel):
    step: str | None = None
    context: Context
    classified_category: ClassifierResult | None = None
    suggested_topics: TopicSuggesterResult | None = None
    summarized_actions: ActionSummarizerResult | None = None
    fulfilled_actions: ActionSummarizerResult | None = None
    inferred_results: InferenceResult | None = None
    selected_topics: TopicSuggesterResult | None = None
    generated_reply_message: MessageGeneratorResult | None = None

    auto_assign_actions: bool = False
    auto_assign_topics: bool = False
    