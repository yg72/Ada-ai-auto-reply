from rexpand_pyutils_file import read_file
from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.classifier import classify_conversation
from nodes.message_generator import generate_message
from nodes.topic_suggester import suggest_topics
from nodes.actions_summarizer import summarize_actions
from nodes.inferencer import inference_results
from langgraph.graph import START, END, StateGraph


CATEGORIES = [Category(**category) for category in read_file("/Users/yiwengeng/Documents/Tuilink/yiwen-auto-reply/input/categories.json")]
EXTENDED_CATEGORY_LOOKUP = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("/Users/yiwengeng/Documents/Tuilink/yiwen-auto-reply/input/categories.json")
}

# Node definitions
def classifier(state: State) -> State:
    if state.classified_category is None:
        state.classified_category = classify_conversation(
            state.context, CATEGORIES, dry_run=False
        )
    return state

def actions_summarizer(state: State) -> State:
    state.summarized_actions = summarize_actions(
        state.context, state.classified_category, dry_run=False
    )
    state.step = "todo: human action required"
    return state

def inferencer(state: State) -> State:
    if state.inferred_results is None:
        state.inferred_results = inference_results(
            state.context, state.classified_category, state.summarized_actions, dry_run=False
        )
    return state

def topic_suggester(state: State) -> State:
    state.suggested_topics = suggest_topics(
        state.context, state.classified_category, state.inferred_results, dry_run=False
    )
    state.step = "next: select topics"
    return state

def message_generater(state: State) -> State:
    state.generated_reply_message = generate_message(
        state.context,
        state.classified_category,
        state.selected_topics,
        dry_run=False,
    )
    state.step = "end: reply generated"
    return state

def no_reply(state: State) -> State:
    state.step = "end: no reply needed"
    return state

def auto_assign_actions(state: State) -> State:
    if state.auto_assign_actions:
        state = state.model_copy(update={"fulfilled_actions": state.summarized_actions or state.fulfilled_actions})
    return state


def auto_assign_topics(state: State) -> State:
    if state.auto_assign_topics:
        state = state.model_copy(update={"selected_topics": state.suggested_topics or state.selected_topics})
    return state


# edge definitions
def route_after_classify(state: State) -> str:
    ext_cat = EXTENDED_CATEGORY_LOOKUP[state.classified_category.category]
    if ext_cat.human_action_required:
        return "summarize_actions"
    else:
        if ext_cat.reply_needed:
            return "suggest_topics"
        else:
            return "no_reply_needed"

def route_after_summarize(state: State) -> str:
    if state.fulfilled_actions:  # Do I need an indicator that human action is received? we don't always want to go infer
        return "infer"
    else:
        return END

def route_after_suggest(state: State) -> str:
    if state.selected_topics:
        return "generate_message"
    else:
        return END


def orchestrate(state: State): # -> State:

    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("classify", classifier)
    workflow.add_node("summarize_actions", actions_summarizer)
    workflow.add_node("auto_assign_actions", auto_assign_actions)
    workflow.add_node("infer", inferencer)
    workflow.add_node("suggest_topics", topic_suggester)
    workflow.add_node("auto_assign_topics", auto_assign_topics)
    workflow.add_node("generate_message", message_generater)
    workflow.add_node("no_reply_needed", no_reply)

    # Add edges
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges("classify", route_after_classify)
    # After summarize_actions → 
    workflow.add_edge("summarize_actions", "auto_assign_actions")
    workflow.add_conditional_edges("auto_assign_actions", route_after_summarize)
    # After infer → suggest_topics
    workflow.add_edge("infer", "suggest_topics")
    # After suggest_topics → generate_message if topics are selected
    workflow.add_edge("suggest_topics", "auto_assign_topics")
    workflow.add_conditional_edges("auto_assign_topics", route_after_suggest) 
    # End edges
    workflow.add_edge("generate_message", END)
    workflow.add_edge("no_reply_needed", END)

    app = workflow.compile()


    return app
