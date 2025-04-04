import streamlit as st

def render_history():
    if st.session_state.history:
        st.markdown("#### Previous Questions & Answers")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.subheader(f"Question: {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                
                score_percentage = int(item['score'] * 100)
                st.progress(item['score'])
                st.caption(f"Confidence: {score_percentage}%")
                
                cols = st.columns(2)
                cols[0].caption(f"Model: {item['model']}")
                cols[1].caption(f"Time: {item['timestamp']}")
                
                st.divider()
    else:
        st.info("You haven't asked any questions yet.")
        if st.button("Go to Home", use_container_width=False):
            st.session_state.page = 'home' 