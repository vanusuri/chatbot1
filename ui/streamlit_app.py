import uuid
import pandas as pd
import streamlit as st

from app.orchestrator import Orchestrator
from app.db.dao import init_db, get_all_tickets, get_recent_logs
from app.logs.logger import logger
from app.rag.ingest import build_support_doc_index


def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def main() -> None:
    st.set_page_config(page_title="Banking Support AI (RAG)", layout="wide")
    st.title("Banking Customer Support AI – Multi-Agent + RAG")

    init_db()
    orchestrator = Orchestrator()

    tab_chat, tab_trace, tab_tickets, tab_metrics = st.tabs(
        ["Chat", "Agent Trace", "Tickets & History", "RAG & Metrics"]
    )

    with tab_chat:
        session_id = ensure_session_id()
        st.subheader("Chat with the AI Agent")
        customer_name = st.text_input(
            "Customer name (optional):", value="", key="customer_name_input"
        )
        message = st.text_area("Enter your message:", key="message_input")

        if st.button("Submit", type="primary"):
            if not message.strip():
                st.warning("Please enter a message.")
            else:
                logger.info("Submitting message from Streamlit UI")
                result = orchestrator.handle_message(
                    message=message,
                    session_id=session_id,
                    customer_name=customer_name or None,
                )
                st.markdown("### Response")
                st.write(result["response"])

                with st.expander("Details"):
                    st.json(result)

    with tab_trace:
        st.subheader("Recent Logs / Agent Trace (simplified view)")
        logs = get_recent_logs(limit=50)
        if logs:
            data = [
                {
                    "timestamp": log.timestamp,
                    "session_id": log.session_id,
                    "classifier": log.classifier,
                    "routed_agent": log.routed_agent,
                    "ticket_number": log.ticket_number,
                    "success": log.success,
                }
                for log in logs
            ]
            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.info("No logs yet.")

    with tab_tickets:
        st.subheader("Support Tickets")
        tickets = get_all_tickets(limit=200)
        if tickets:
            data = [
                {
                    "ticket_number": t.ticket_number,
                    "customer_name": t.customer_name,
                    "status": t.status,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                    "channel": t.channel,
                    "message": t.message,
                }
                for t in tickets
            ]
            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.info("No tickets found.")

    with tab_metrics:
        st.subheader("RAG & Simple Metrics")

        st.markdown("#### Rebuild Support Document Index")
        st.write(
            "Place `.txt` or `.md` files under the `knowledge_base/` folder, "
            "then click the button below to rebuild the RAG index."
        )

        if st.button("Rebuild RAG Index"):
            with st.spinner("Building support document index (this may take a while)..."):
                try:
                    build_support_doc_index()
                    st.success("Support document index rebuilt successfully.")
                except Exception as e:  # noqa: BLE001
                    logger.exception("Error rebuilding RAG index")
                    st.error(f"Error rebuilding RAG index: {e}")

        logs = get_recent_logs(limit=200)
        st.markdown("#### Basic Event Metrics")
        if logs:
            total = len(logs)
            success_count = sum(1 for l in logs if l.success)
            st.metric("Total events", total)
            st.metric("Successful events", success_count)
        else:
            st.info("No log data available for metrics yet.")


if __name__ == "__main__":
    main()
