import pandas as pd
import streamlit as st


def render_schedule_tab(*, generate_schedule, get_last_generated_schedule):
    st.header("Generated Schedule")

    if st.button("Generate Schedule"):
        with st.spinner("Generating schedule..."):
            warnings = generate_schedule(
                st.session_state.availability,
                st.session_state.start_date,
                export_to_excel=False,
            )
            st.session_state.warnings = warnings
            st.session_state.generated_schedule = pd.DataFrame(get_last_generated_schedule())
            if not warnings:
                st.toast("✅ Schedule generated successfully!")
            else:
                st.toast(f"⚠️ Schedule generated with {len(warnings)} warnings.")

    # --- Display Generated Schedule ---
    if st.session_state.generated_schedule is not None:
        # Display warnings if any
        if st.session_state.warnings:
            st.warning("The following issues were found during schedule generation:")
            for warning in st.session_state.warnings:
                st.write(f"- {warning}")

        st.dataframe(st.session_state.generated_schedule)
