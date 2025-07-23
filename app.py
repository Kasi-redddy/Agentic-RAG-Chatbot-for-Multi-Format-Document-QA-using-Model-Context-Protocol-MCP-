import streamlit as st
import os
import json # For pretty printing JSON
from agents import CoordinatorAgent # Import the CoordinatorAgent

# Streamlit UI
def main():
    st.set_page_config(page_title="Agentic RAG Chatbot with MCP", layout="wide")
    st.title("Agentic RAG Chatbot with MCP")
    st.markdown("Upload documents (PDF, CSV, DOCX, PPTX, TXT, Markdown) and ask questions based on their content.")

    # Initialize session state
    if "coordinator" not in st.session_state:
        st.session_state.coordinator = CoordinatorAgent()
        st.session_state.uploaded_file_names = set() 
        st.session_state.chat_history = [] 
    
    # Ensure uploads directory exists (though not strictly used in current in-memory setup)
    if "uploads_dir_checked" not in st.session_state:
        os.makedirs("uploads", exist_ok=True)
        st.session_state.uploads_dir_checked = True

    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "csv", "docx", "txt", "md", "pptx"],
        help="Supported formats: PDF, CSV, DOCX, PPTX, TXT, Markdown"
    )

    if uploaded_files:
        files_to_process = [f for f in uploaded_files if f.name not in st.session_state.uploaded_file_names]
        
        if files_to_process:
            with st.spinner("Processing documents... This may take a moment."):
                processed_count = 0
                error_messages = []
                for uploaded_file in files_to_process:
                    try:
                        content = uploaded_file.read()
                        st.session_state.coordinator.process_document_upload(uploaded_file.name, content)
                        st.session_state.uploaded_file_names.add(uploaded_file.name)
                        processed_count += 1
                        # logger.info(f"Successfully processed new file: {uploaded_file.name}") # Logger in agents.py now
                    except Exception as e:
                        error_messages.append(f"Error processing {uploaded_file.name}: {str(e)}")
                        # logger.error(f"File upload error for {uploaded_file.name}: {str(e)}", exc_info=True) # Logger in agents.py now
                
                if processed_count > 0:
                    st.success(f"Successfully processed {processed_count} new file(s)!")
                if error_messages:
                    for msg in error_messages:
                        st.error(msg)
        else:
            st.info("No new files to process or all selected files have been processed already.")

    st.subheader("Currently Uploaded Documents:")
    if st.session_state.uploaded_file_names:
        for fname in st.session_state.uploaded_file_names:
            st.markdown(f"- {fname}")
    else:
        st.info("No documents uploaded yet.")

    # Query input
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question (e.g., What KPIs were tracked in Q1?)", key="query_input")
    
    if st.button("Submit Query", disabled=not query or not st.session_state.uploaded_file_names):
        if not st.session_state.uploaded_file_names:
            st.warning("Please upload documents before asking a question.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    answer, retrieval_mcp_output, original_sources_info = st.session_state.coordinator.handle_query(query)
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer,
                        "retrieval_mcp_output": retrieval_mcp_output,
                        "original_sources_info": original_sources_info 
                    })
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    # logger.error(f"Query processing error: {str(e)}", exc_info=True) # Logger in agents.py now

    # Display chat history (multi-turn)
    st.subheader("Chat History")
    for chat_entry in reversed(st.session_state.chat_history):
        st.markdown(f"**Question:** {chat_entry['question']}")
        st.markdown(f"**Answer:** {chat_entry['answer']}")
        
        if chat_entry['retrieval_mcp_output']:
            st.markdown("**MCP Retrieval Result:**")
            st.json(chat_entry['retrieval_mcp_output'])
            
            st.markdown("**Simple Sources from Context:**")
            for i, source_detail in enumerate(chat_entry['original_sources_info'], 1):
                st.markdown(f"  - {source_detail['source_info']}")
        
        st.markdown("---")

if __name__ == "__main__":
    main()