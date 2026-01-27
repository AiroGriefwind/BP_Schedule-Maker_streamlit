import streamlit as st


def render_schedule_tab():
    # --- Display Generated Schedule ---
    if st.session_state.generated_schedule is not None:
        st.header("Generated Schedule")

        # Display warnings if any
        if st.session_state.warnings:
            st.warning("The following issues were found during schedule generation:")
            for warning in st.session_state.warnings:
                st.write(f"- {warning}")

        st.dataframe(st.session_state.generated_schedule)
