##-- To run the streamlit app, run the following command in the terminal: streamlit run streamlit_app.py
# pip install pandas selfcheckgpt sentence_transformers spacy streamlit torch

import streamlit as st
import utils
import pandas as pd
from time import time
import re

# Streamlit app layout
st.title('Hallucination Mitigation')

# Text input
user_input = st.text_input("Enter your text:")
#st.image('selfcheck_llm.png', caption='LLM prompted Score')     ###---- Comment out if no image


if user_input:

    prompt = user_input

    output, sampled_passages = utils.get_output_and_samples(prompt)

    # # NLI score
    # self_similarity_score = utils.get_self_check_nli(output,sampled_passages)
    # st.write(self_similarity_score)
    # st.write("Actual output: ", output)
    # st.write("**Sampled passages:**")
    # for i, passage in enumerate(sampled_passages):
    #     st.write(f"Sample {i+1}: {passage}")
    # LLM score
    
    start=time()
    self_similarity_score = utils.llm_evaluate2(output,sampled_passages) #use llm_evaluate or llm_evaluate2

    print('self_similarity_score:',self_similarity_score)
    print(type(self_similarity_score))
    try:
        self_similarity_score = float(self_similarity_score)
    except ValueError:
    # Extract number with decimal point using a capturing group
        match = re.search(r'\d+\.\d+', self_similarity_score) 
        if match:
            self_similarity_score = float(match.group())  # Extract the entire matched group
            # If llm_evaluate instead of llm_evaluate2 is used, then add line self_similarity_score=1-self_similarity_score
        else:
            # Handle cases where no number with decimal point is found (optional)
            self_similarity_score = 0.5  # Or raise an error

    end = time()

    # Display the output
    print(self_similarity_score)
    st.write("Score:",self_similarity_score)
    st.write("**LLM output:**")
    threshold = 0.5
    if self_similarity_score < threshold:           ####----- Change the threshold value as needed
        st.write(output)
        st.write("**Sampled passages:**")
        for i, passage in enumerate(sampled_passages):
            st.write(f"Sample {i+1}: {passage}")
    else:
        st.write(f"I'm sorry, but I don't have the specific information required to answer your question accurately. Self-similarity score {self_similarity_score} is above the threshold {threshold}.\nNOTE: score closer to 1 indicates higher chance of hallucination")
        st.write("**Actual output:** ", output)
        st.write("**Sampled passages:**")
        for i, passage in enumerate(sampled_passages):
            st.write(f"Sample {i+1}: {passage}")
    # add image in streamlit
    st.image('llm_prompted_Score.png', caption='LLM prompted Score')   ####----- Comment out if no image

# Hallucination score = 1- self_similarity_score
