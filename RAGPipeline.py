#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install langchain transformers faiss-cpu datasets ragas nltk rouge-score sentence_transformers langchain_community')


# In[1]:


import sys
print(sys.executable)


# In[2]:


from datasets import load_dataset

input_data = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")


# In[3]:


input_data


# In[4]:


import numpy as np

len(np.unique(input_data['passages']['id']))


# In[6]:


new_contexts=input_data['passages']['passage']


# In[9]:


# Check for empty strings
has_empty_strings = '' in new_contexts
print(has_empty_strings)  # Output: True


# In[10]:


from sentence_transformers import SentenceTransformer

# Load pre-trained model
embeddingmodel = SentenceTransformer('paraphrase-MiniLM-L3-v2')


# In[11]:


embeddingmodel


# In[12]:


# # Vectorize the context data
context_embeddings = embeddingmodel.encode(new_contexts, convert_to_tensor=True)


# In[ ]:


### Use following function if need to use same tokenizer as used in text-generation pipeline below.####


# In[ ]:


import torch
from transformers import AutoTokenizer, AutoModel

def get_last_hidden_state_embeddings(texts, model_name='distilgpt2'):
    """
    Captures the last hidden state as embeddings for a given list of texts.

    Args:
    texts (list of str): List of text for which we want embeddings.
    model_name (str): Name of the pre-trained model to use (default is 'distilgpt2').

    Returns:
    torch.Tensor: Tensor containing embeddings for each text.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Encode the texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Take the mean of the last hidden state to get a single embedding per input
    mean_embeddings = torch.mean(embeddings, dim=1)
    
    return mean_embeddings

context_embeddings = get_last_hidden_state_embeddings(new_contexts)


# In[13]:


context_embeddings.shape


# In[14]:


import faiss
import numpy as np

# Create a FAISS index
index = faiss.IndexFlatL2(context_embeddings.shape[1])


# In[17]:


index.add(np.array(context_embeddings))


# In[18]:


def retrieve_documents(query, k=5):
    query_embedding = embeddingmodel.encode([query], convert_to_tensor=True)
    distances, indices = index.search(np.array(query_embedding), k)
    return [new_contexts[i] for i in indices[0]]


# In[19]:


from langchain import LLMChain, PromptTemplate
from transformers import pipeline


# In[22]:


from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# Load a pre-trained language model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
llmmodel = AutoModelForCausalLM.from_pretrained('distilgpt2')


# In[23]:


# Define a prompt template
template = PromptTemplate(
    input_variables=["query", "documents"],
    template="Query: {query}\nDocuments: {documents}\nAnswer:"
)


# In[25]:


from langchain.llms import HuggingFacePipeline
from langchain import LLMChain, PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# In[27]:


hf_pipeline = pipeline('text-generation', model=llmmodel, tokenizer=tokenizer,max_length=1024,max_new_tokens=100)


# In[28]:


# Wrap the Hugging Face pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# In[29]:


# Create a LangChain LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=template
)


# In[30]:


llm_chain


# In[42]:


### manually truncating input length. If its higher than 924, 
## truncating it to take to the extent it can accomodate 924 tokens. Leaving 100 for output tokens.
### 1024 is the max limit for the model selected.
def truncate_input(query, documents, max_length=1024, max_new_tokens=100):
    combined_input = f"Query: {query}\nDocuments: {documents}\nAnswer:"
    input_tokens = tokenizer.encode(combined_input)
    if len(input_tokens) > (max_length - max_new_tokens):
        truncated_docs = documents[:max_length - len(tokenizer.encode(f"Query: {query}\nAnswer:")) - max_new_tokens]
        combined_input = f"{truncated_docs}"
        return combined_input
    return documents


# In[45]:


### using context documents as input ###
def generate_response(query,k):
    documents = retrieve_documents(query,k)
    retrieved_docs_flatten=' '.join(documents)
    truncated_docs=truncate_input(query, retrieved_docs_flatten, max_length=1024, max_new_tokens=100)
    response = llm_chain.run(query=query, documents=truncated_docs,max_length=1024,max_new_tokens=100,temperature=0.7,top_k=50,top_p=0.9)
    return response


# In[47]:


### without using context documents as input ###
def generate_response_without_docs(query,k):
    truncated_docs=''
    response = llm_chain.run(query=query, documents=truncated_docs,max_length=1024,max_new_tokens=100,temperature=0.7,top_k=50,top_p=0.9)
    return response


# In[54]:


def extract_answer_only(response):
    """
    Extracts the portion of the response starting after 'Answer:'.

    Args:
    response (str): The full response text.

    Returns:
    str: The extracted answer text or a message if 'Answer:' is not found.
    """
    # Find the starting position of "Answer:"
    start_index = response.find("Answer:")
    
    # Extract the text starting after "Answer:"
    if start_index != -1:
        answer_text = response[start_index + len("Answer:"):].strip()
        return answer_text
    else:
        return "The keyword 'Answer:' was not found in the response."


# In[51]:


#### question answer pair dataset to check and compare the output when we use context versus without
from datasets import load_dataset

input_output_pairs = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")


# In[55]:


input_output_pairs['test']


# In[67]:


## example question
query=input_output_pairs['test']['question'][10]
query


# In[68]:


## example answer
answer=input_output_pairs['test']['answer'][10]
answer


# In[69]:


import random

# Generate a list of 100 random integers between 0 and 4719
random_list = [random.randint(0, 4719) for _ in range(100)]


# In[71]:


responses_with_contexts=[]
responses_without_contexts=[]
original_answer = []


# In[72]:


#### generate 100 responses
for i in random_list:
    query=input_output_pairs['test']['question'][i]
    answer = input_output_pairs['test']['answer'][i]
    response=extract_answer_only(generate_response(query,5))
    responses_with_contexts.append(response)
    response=extract_answer_only(generate_response_without_docs(query,5))
    responses_without_contexts.append(response)
    original_answer.append(answer)


# In[75]:


from nltk.translate.bleu_score import sentence_bleu

# Calculate BLEU scores
bleu_scores = [sentence_bleu([human], llm) for human, llm in zip(original_answer, responses_with_contexts)]

# Average BLEU score
average_bleu_with_contextx = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score RAG: {average_bleu_with_contextx}")


# In[76]:


# Calculate BLEU scores
bleu_scores = [sentence_bleu([human], llm) for human, llm in zip(original_answer, responses_without_contexts)]

# Average BLEU score
average_bleu_with_contextx = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score w/oRAG: {average_bleu_with_contextx}")


# In[78]:


##### as expected BLUE score with RAG is higher (~0.146) as compared to without RAG (~0.13)
## Although both scores are low, but this is expected due to the simplifications assumed


# In[80]:


from rouge_score import rouge_scorer

# Initialize the scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Calculate ROUGE scores
rouge_scores = [scorer.score(human, llm) for human, llm in zip(original_answer, responses_with_contexts)]

# Print average ROUGE scores
average_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
average_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
average_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)

print(f"Average ROUGE-1 score: {average_rouge1}")
print(f"Average ROUGE-2 score: {average_rouge2}")
print(f"Average ROUGE-L score: {average_rougeL}") 


# In[ ]:





# In[81]:


# Calculate ROUGE scores
rouge_scores = [scorer.score(human, llm) for human, llm in zip(original_answer, responses_without_contexts)]

# Print average ROUGE scores
average_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
average_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
average_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)

print(f"Average ROUGE-1 score w/o RAG: {average_rouge1}")
print(f"Average ROUGE-2 score w/o RAG: {average_rouge2}")
print(f"Average ROUGE-L score w/o RAG: {average_rougeL}") 


# In[87]:


######### Using Rogue score gving higher values to without contexts which is a counter-intuitive
### might be because it considers exact keyword pairs as inputs using n-grams approach


# In[92]:


## Answer Semantic Similarity


# In[89]:


# Compute embeddings for all responses
from sentence_transformers import util
llm_with_contexts_embeddings = embeddingmodel.encode(responses_with_contexts, convert_to_tensor=True)
human_embeddings = embeddingmodel.encode(original_answer, convert_to_tensor=True)

# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity: {average_cosine_similarity}")


# In[90]:


# Compute embeddings for all responses w/o contexts
from sentence_transformers import util

llm_with_contexts_embeddings = embeddingmodel.encode(responses_without_contexts, convert_to_tensor=True)
human_embeddings = embeddingmodel.encode(original_answer, convert_to_tensor=True)

# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity w/o RAG: {average_cosine_similarity}")


# In[91]:


#### here also cosine similarity is higher for without RAG


# In[94]:


#### Perplexity ##########


# In[98]:


import torch

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = llmmodel(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Calculate perplexity for each response
llm_with_contexts_perplexities = [calculate_perplexity(response) for response in responses_with_contexts]
llm_without_contexts_perplexities = [calculate_perplexity(response) for response in responses_without_contexts]

# Calculate average perplexity
average_llm_with_context_perplexity = sum(llm_with_contexts_perplexities) / len(llm_with_contexts_perplexities)
average_llm_without_context_perplexity = sum(llm_without_contexts_perplexities) / len(llm_without_contexts_perplexities)

print(f"Average LLM Perplexity with RAG: {average_llm_with_context_perplexity}")
print(f"Average LLM Perplexity w/o RAG: {average_llm_without_context_perplexity}")


# In[103]:


### here also results show better with/o RAG might be due to data truncations and model assumed.


# In[ ]:


##### check semantic similarity with question ####


# In[110]:


original_question=[]


# In[111]:


for i in random_list:
    query=input_output_pairs['test']['question'][i]
    original_question.append(query)


# In[120]:


# Compute embeddings for all responses
from sentence_transformers import util
llm_with_contexts_embeddings = embeddingmodel.encode(responses_with_contexts, convert_to_tensor=True)
human_embeddings = embeddingmodel.encode(original_question, convert_to_tensor=True)

# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity of responses with questions RAG: {average_cosine_similarity}")


# In[121]:


# Compute embeddings for all responses w/o contexts
from sentence_transformers import util

llm_with_contexts_embeddings = embeddingmodel.encode(responses_without_contexts, convert_to_tensor=True)
human_embeddings = embeddingmodel.encode(original_question, convert_to_tensor=True)

# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity of responses with questions w/o RAG: {average_cosine_similarity}")


# In[128]:


##### might be we are using different embeddings model to evaluate similarity ####


# In[138]:


tokenizer.pad_token = tokenizer.eos_token


# In[ ]:





# In[139]:


#### using the embeddings from distilgpt2 model
def get_embeddings_gpt2(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    # inputs = tokenizer(texts, return_tensors='pt',truncation=True)
    with torch.no_grad():
        outputs = llmmodel(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    # Use the last hidden state as embeddings
    embeddings = hidden_states[-1].mean(dim=1)  # Mean pooling over the sequence length
    return embeddings


# In[140]:


# Compute embeddings for all responses
from sentence_transformers import util
llm_with_contexts_embeddings = get_embeddings_gpt2(responses_with_contexts)
human_embeddings = get_embeddings_gpt2(original_question)

# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity of responses with questions RAG: {average_cosine_similarity}")


# In[141]:


# Compute embeddings for all responses
from sentence_transformers import util
llm_with_contexts_embeddings = get_embeddings_gpt2(responses_without_contexts)
human_embeddings = get_embeddings_gpt2(original_question)

# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity of responses with questions w/oRAG: {average_cosine_similarity}")


# In[145]:


#### here we can see cosine similarity higher with RAG as compared to without RAG


# In[146]:


# Recompute embeddings for all responses with gpt2 model embeddings

llm_with_contexts_embeddings = get_embeddings_gpt2(responses_with_contexts)
human_embeddings = get_embeddings_gpt2(original_answer)


# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity: {average_cosine_similarity}")


# In[147]:


# Recompute embeddings for all responses with gpt2 model embeddings

llm_with_contexts_embeddings = get_embeddings_gpt2(responses_without_contexts)
human_embeddings = get_embeddings_gpt2(original_answer)


# Compute cosine similarities
cosine_similarities = [util.pytorch_cos_sim(llm, human).item() for llm, human in zip(llm_with_contexts_embeddings, human_embeddings)]

# Calculate average cosine similarity
average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
print(f"Average Cosine Similarity without RAG: {average_cosine_similarity}")


# In[148]:


#### performance is slightly lower without RAG...


# In[150]:


#### some other metrics from ragas library like answer relevancy, faithfulness
### but that requires Open AI key. Hence did not explore further.
### these are good to explore though as a next step as these are specifically relevant for RAG


# In[105]:


from ragas.metrics import faithfulness


# In[106]:


from ragas import evaluate


# In[108]:


from datasets import Dataset 
from ragas.metrics import faithfulness,answer_relevancy
from ragas import evaluate

data_samples_foo = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
}
dataset_foo = Dataset.from_dict(data_samples_foo)
score = evaluate(dataset_foo,metrics=[answer_relevancy])
score.to_pandas()


# In[ ]:




