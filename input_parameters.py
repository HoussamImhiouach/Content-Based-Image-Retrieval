# user_input.py
import streamlit as st

def user_input_parameters():
    with st.sidebar:
        st.header(":red[Choose your image parameteres]")
        st.write("Descriptor")
        descriptor = st.radio("Choose a descriptor", ["GLCM", "Bitdesc"])
        st.write("Distance")
        distance = st.radio("Choose a distance measure", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"])
        number = st.slider("Select the number of images to display", min_value=1, max_value=200, value=5)

        # Return the selected values
        return {
            "Descriptor": descriptor,
            "Distance": distance,
            "Number": number
        }


