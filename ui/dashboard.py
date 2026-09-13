import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.agent import run_agent
from memory.graph_memory import get_facts_about

st.set_page_config(page_title="Self-Learning Agent Dashboard", layout="wide")

st.title("🤖 Self-Learning Multi-Agent System")
st.caption("Agentic AI with Reflexion, Dual Memory, and Hybrid RAG")

# Sidebar: Knowledge Graph facts
with st.sidebar:
    st.header("🧠 Knowledge Graph")
    st.write("Facts the system knows about you:")
    try:
        facts = get_facts_about("User")
        if facts:
            for fact in facts:
                st.markdown(f"- {fact}")
        else:
            st.write("No facts stored yet.")
    except Exception as e:
        st.warning("Knowledge graph unavailable.")

    st.divider()
    st.header("ℹ️ System Status")
    st.markdown("✅ Vector Memory (Qdrant)")
    st.markdown("✅ Knowledge Graph (Neo4j)")
    st.markdown("✅ Hybrid RAG (Documents)")
    st.markdown("✅ Reflexion Loop")

# Main chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Chat with the Agent")

task = st.text_input("Ask the agent something:", placeholder="e.g. How much does the product cost?")

if st.button("Run Agent") and task:
    with st.spinner("Agent is thinking (retrieving memory, searching documents, reflecting)..."):
        result = run_agent(task)
    st.session_state.chat_history.append((task, result))

# Display chat history
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Agent:** {a}")
    st.divider()