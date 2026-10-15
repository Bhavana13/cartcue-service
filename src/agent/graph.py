import logging
from langgraph.graph import StateGraph, END
from src.agent.state import NovaAgentState
from src.agent.nodes import retrieve_evidence_node, classify_cause_node, generate_explanation_node

# Configure structured logging for the agent workflow graph
logger = logging.getLogger("cartcue-agent-graph")

def build_nova_graph():
    logger.info("Building Nova agent state workflow graph...")
    workflow = StateGraph(NovaAgentState)

    # Add workflow nodes
    workflow.add_node("retrieve_evidence", retrieve_evidence_node)
    workflow.add_node("classify_cause", classify_cause_node)
    workflow.add_node("generate_explanation", generate_explanation_node)

    # Set up edges and execution flow
    workflow.set_entry_point("retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "classify_cause")
    workflow.add_edge("classify_cause", "generate_explanation")
    workflow.add_edge("generate_explanation", END)

    compiled_app = workflow.compile()
    logger.info("Nova agent graph successfully compiled and ready for invocation.")
    return compiled_app

nova_app = build_nova_graph()