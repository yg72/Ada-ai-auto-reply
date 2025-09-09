from rexpand_pyutils_file import read_file

from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.classifier import classify_conversation
from nodes.message_generator import generate_message
from nodes.topic_suggester import suggest_topics
from nodes.actions_summarizer import summarize_actions
from nodes.inferencer import inference_results


CATEGORIES = [Category(**category) for category in read_file("./input/categories.json")]
EXTENDED_CATEGORY_LOOKUP = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("./input/categories.json")
}


def orchestrate(
    state: State,
) -> State:
    categories: list[Category] = CATEGORIES
    extended_category_lookup: dict[str, ExtendedCategory] = EXTENDED_CATEGORY_LOOKUP

    # Run classifier to get the classified category
    if state.classified_category is None:
        state.classified_category = classify_conversation(
            state.context, categories, dry_run=False
        )

    # Get the extended category with post-processing indicators
    extended_category = extended_category_lookup[state.classified_category.category]

    # If human action is required, run the actions summarizer and prompt the human to take action
    if extended_category.human_action_required:
        # Add actions summarizer
        if state.summarized_actions is None:
            state.summarized_actions = summarize_actions(
                state.context, state.classified_category, dry_run=False
            )

        if state.auto_assign_actions:
            state.fulfilled_actions = state.summarized_actions
        
        if state.fulfilled_actions is None:
            state.step = "next: take actions"
            return state
        else:
            if state.inferred_results is None:
                state.inferred_results = inference_results(
                    state.context, state.classified_category, state.summarized_actions, dry_run=False)

            if state.suggested_topics is None:
                state.suggested_topics = suggest_topics(
                    state.context, state.classified_category, state.inferred_results, dry_run=False)
            if state.auto_assign_topics:
                state.selected_topics = state.suggested_topics
            if state.selected_topics is None:
                state.step = "next: select topics"
                return state
            # If topics are selected, generate the reply message
            else:
                if state.generated_reply_message is None:
                    state.generated_reply_message = generate_message(
                        state.context,
                        state.classified_category,
                        state.selected_topics,
                        dry_run=False,
                    )

                state.step = "end: reply generated"
                return state
            
    else:
        # If reply is needed, check if topics are selected
        if extended_category.reply_needed:
            # If no topics are selected, prompt the human to select suggestedtopics
            if state.suggested_topics is None:
                state.suggested_topics = suggest_topics(
                    state.context, state.classified_category, state.inferred_results, dry_run=False)
            if state.auto_assign_topics:
                state.selected_topics = state.suggested_topics
            if state.selected_topics is None:
                state.step = "next: select topics"
                return state
            # If topics are selected, generate the reply message
            else:
                if state.generated_reply_message is None:
                    state.generated_reply_message = generate_message(
                        state.context,
                        state.classified_category,
                        state.selected_topics,
                        dry_run=False,
                    )

                state.step = "end: reply generated"
                return state

        # If no reply is needed, return the current state
        else:
            state.step = "end: no reply needed"
            return state
