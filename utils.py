from sentence_transformers import SentenceTransformer
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNLI
import pandas as pd
import torch 
import spacy

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    mps_device='cpu'

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

##------------------option1
def consistency_check_LLM(sentence, context):
    response = eli_chat(
    model="llama",
    messages=[
        {
        "role": "system",
        "content": f"You will be provided with a sentence, and your task is to classify whether the provided context supports the sentence. Your answer must be one of the following: 'Yes', 'No'.Context: {context}. Sentence:" #Give answer as No even if slight inconsistency
        },
        {
        "role": "user",
        "content": sentence,
        }
    ],
    temperature=0,
    max_new_tokens=1024,
    )
    return response
def llm_evaluate2(sentences,sampled_passages):
    # sentence splitting with spacy
    nlp = spacy.load("en_core_web_sm")
    sentences = [sent for sent in nlp(sentences).sents] # List[spacy.tokens.span.Span]
    sentences = [sent.text.strip() for sent in sentences if len(sent) > 2] #>3
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
    
#use llm_evaluate or llm_evaluate2

##------------------option2
def llm_evaluate(sentences, sampled_passages):
    system_message = "You will be provided with a text passage \
                and your task is to rate the consistency of that text to \
                that of the provided context. Your answer must be only \
                a number between 0.0 and 1.0 rounded to the nearest two \
                decimal places where 0.0 represents no consistency and \
                1.0 represents perfect consistency and similarity. Only give final answer in decimal format. Give Score below 0.5 even if slight inaccuracies"
    prompt = f"""
                Text passage: {sentences}. \n\n \
                Context: {sampled_passages[0]} \n\n \
                {sampled_passages[1]} \n\n \
                {sampled_passages[2]}."""

    result = eli_chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_new_tokens=100,
    )
    completion = result

    return completion
    
def generate_3_samples(prompt):
    system_message = "Give short answers."
    sampled_passages = []
    for i in range(1,4):
        result = eli_chat(
            model="llama3", #llama2
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_new_tokens=200,
        )
        globals()[f'sample_{i}'] = result.lstrip('\n')
        sampled_passages.append(globals()[f'sample_{i}'])
    return sampled_passages

def get_output_and_samples(prompt):
    system_message = "Give short answers."
    result = eli_chat(
        model="llama3", #llama2
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_new_tokens=100,
    )
    output = result

    sampled_passages = generate_3_samples(prompt)
    return output, sampled_passages



#--------
# CAN USE get_bertscore() and get_self_check_nli() functions to get the hallucination score and probability of contradiction respectively instead of LLM prompt if needed
def get_bertscore(output, sampled_passages):
    # spacy sentence tokenization
    sentences = [sent.text.strip() for sent in nlp(output).sents] 
    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences = sentences, # list of sentences
        sampled_passages = sampled_passages, # list of sampled passages
    )
    df = pd.DataFrame({
    'Sentence Number': range(1, len(sent_scores_bertscore) + 1),
    'Hallucination Score': sent_scores_bertscore
    })
    return df

def get_self_check_nli(output, sampled_passages):
    # spacy sentence tokenization
    sentences = [sent.text.strip() for sent in nlp(output).sents] 
    selfcheck_nli = SelfCheckNLI(device=mps_device) # set device to 'cuda' if GPU is available
    sent_scores_nli = selfcheck_nli.predict(
        sentences = sentences, # list of sentences
        sampled_passages = sampled_passages, # list of sampled passages
    )
    df = pd.DataFrame({
    'Sentence Number': range(1, len(sent_scores_nli) + 1),
    'Probability of Contradiction': sent_scores_nli
    })
    return sent_scores_nli[0],df
