import streamlit as st
import requests


# Page configuration
st.set_page_config(
    page_title="AI IT Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)


# Title
st.title("🎫 AI IT Support Ticket Classifier")

st.write(
    "Enter an IT support ticket to predict its queue, priority, "
    "type, and find similar historical tickets."
)


# Ticket input
st.subheader("Ticket Details")

subject = st.text_input(
    "Subject",
    placeholder="Example: Application is completely down"
)

body = st.text_area(
    "Description",
    placeholder="Describe the issue in detail...",
    height=180
)

top_n = st.slider(
    "Number of similar tickets",
    min_value=1,
    max_value=10,
    value=5
)


# Predict button
if st.button("Predict Ticket", type="primary"):

    if not subject.strip() or not body.strip():
        st.warning("Please enter both the subject and description.")

    else:

        payload = {
            "subject": subject,
            "body": body,
            "top_n": top_n
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]
                margins = result["margins"]
                review = result["manual_review"]
                similar_tickets = result["similar_tickets"]


                # Prediction section
                st.subheader("Predictions")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Queue",
                        prediction["queue"]
                    )

                with col2:
                    st.metric(
                        "Priority",
                        prediction["priority"].upper()
                    )

                with col3:
                    st.metric(
                        "Type",
                        prediction["type"]
                    )


                # Confidence / review section
                st.subheader("Prediction Review")

                review_col1, review_col2, review_col3 = st.columns(3)

                with review_col1:
                    st.write("**Queue Margin**")
                    st.write(f"{margins['queue']:.3f}")

                    if review["queue"]:
                        st.warning("Manual review recommended")
                    else:
                        st.success("Prediction looks reliable")

                with review_col2:
                    st.write("**Priority Margin**")
                    st.write(f"{margins['priority']:.3f}")

                    if review["priority"]:
                        st.warning("Manual review recommended")
                    else:
                        st.success("Prediction looks reliable")

                with review_col3:
                    st.write("**Type Margin**")
                    st.write(f"{margins['type']:.3f}")

                    if review["type"]:
                        st.warning("Manual review recommended")
                    else:
                        st.success("Prediction looks reliable")


                # Similar tickets
                st.subheader("Similar Historical Tickets")

                for i, ticket in enumerate(similar_tickets, start=1):

                    with st.expander(
                        f"Ticket {i} — Similarity: "
                        f"{ticket['similarity']:.3f}"
                    ):

                        st.write(ticket["text"])

                        info_col1, info_col2, info_col3 = st.columns(3)

                        with info_col1:
                            st.write(
                                f"**Queue:** {ticket['queue']}"
                            )

                        with info_col2:
                            st.write(
                                f"**Priority:** {ticket['priority']}"
                            )

                        with info_col3:
                            st.write(
                                f"**Type:** {ticket['type']}"
                            )


            else:

                st.error(
                    f"API returned error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure the API is running on port 8000."
            )