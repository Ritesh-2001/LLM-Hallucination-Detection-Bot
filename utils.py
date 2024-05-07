from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNLI
import pandas as pd
import torch 
import spacy
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

# LLM model. This model is used to generate samples for evaluation
model=ChatOllama(model="llama3") #llama3 or llama2 or mistral


def consistency_check_LLM(sentence, context):
    consistency_check_template = """You will be provided with a sentence, and your task is to classify whether the provided context supports the sentence. Your answer must be one of the following: 'Yes', 'No'.Context: {context}. Sentence: {sentence}"""
    consistency_check_prompt = ChatPromptTemplate.from_template(consistency_check_template)

    consistency_check_chain = (
        {"context": context, "sentence": sentence}
        | consistency_check_prompt
        | model.invoke
        | StrOutputParser()
    )
    response = consistency_check_chain.invoke(sentence)
    return response

def llm_evaluate(sentences,sampled_passages):
    # sentence splitting with spacy
    nlp = spacy.load("en_core_web_sm")
    sentences = [sent for sent in nlp(sentences).sents] # List[spacy.tokens.span.Span]
    sentences = [sent.text.strip() for sent in sentences if len(sent) > 3] 
    llm_scores=[]
    for sentence in sentences:
        scores = []
        for context in sampled_passages:
            answer = consistency_check_LLM(sentence, context)
            scores.append(answer)
        print('###############################')
        print(scores)
        llm_scores.append(1 - scores.count('Yes')/len(scores))
    return sum(llm_scores)/len(llm_scores)


def generate_5_samples(prompt):
    sampled_passages = []
    for i in range(1,6):
        result = model.invoke(
            prompt,
            temperature=0.7,
        ).content  # Invoke the model and get the content
        globals()[f'sample_{i}'] = result.lstrip('\n')
        sampled_passages.append(globals()[f'sample_{i}'])
    return sampled_passages

def get_output_and_samples(prompt):
    completion = model.invoke(
    prompt,
    temperature=0.7,
    )
    output = completion.content

    sampled_passages = generate_5_samples(prompt)
    return output, sampled_passages
