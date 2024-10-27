# RAG-Enhanced-QA
This repository contains the code for a Document Retrieval and Question Answering (DRQA) system built using the RAG (Retriever-Augmenter-Generator) approach. This system leverages pre-trained models to:

1. Retrieve relevant passages from a corpus based on the user's query.
2. Generate an answer that incorporates information from the retrieved passages to directly address the user's question.

## Requirements
- Python 3.6+
- Libraries:
  -   langchain
  -   transformers
  -   faiss-cpu
  -   datasets
  -   ragas (optional, for RAG specific metrics)
  -   nltk (optional, for ROUGE score calculation)
  -   rouge_score (optional, for ROUGE score calculation)
  -   sentence-transformers

## Data
This example utilizes the rag-datasets/rag-mini-bioasq dataset for illustrative purposes. You can substitute this with your own dataset adhering to the same format (question-answer-passage). It can also be used in contexts where we don't have human evaluation of generated responses. In those cases, the evaluation metrics used would be different e.g. perplexity which do not require inputs on human evaluation.

## Key Code Snippets and Explanations

## Document Retrieval:

```
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Encode the context passages into embeddings
context_embeddings = embedding_model.encode(new_contexts, convert_to_tensor=True)

# Create a FAISS index for efficient nearest neighbor search
index = faiss.IndexFlatL2(context_embeddings.shape[1])
index.add(np.array(context_embeddings))

# Retrieve top-k most similar passages to the query
def retrieve_documents(query, k=5):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(np.array(query_embedding), k)
    return [new_contexts[i] for i in indices[0]]
```

This code snippet demonstrates how to:

1. Encode context passages and queries into semantic embeddings using a pre-trained sentence transformer.
2. Create a FAISS index to efficiently search for the top-k most similar passages to the query.

## Answer Generation:

```
from langchain import LLMChain, PromptTemplate
from transformers import pipeline

# Load a pre-trained language model
llm_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
hf_pipeline = pipeline('text-generation', model=llm_model, tokenizer=tokenizer)

# Define a prompt template to guide the LLM
template = PromptTemplate(
    input_variables=["query", "documents"],
    template="Query: {query}\nDocuments: {documents}\nAnswer:"
)

# Create an LLMChain to combine the prompt template and LLM
llm_chain = LLMChain(
    llm=HuggingFacePipeline(pipeline=hf_pipeline),
    prompt=template
)

# Generate an answer given a query and retrieved documents
def generate_response(query, k):
    documents = retrieve_documents(query, k)
    response = llm_chain.run(query=query, documents=' '.join(documents))
    return response
```

This code snippet demonstrates how to:

1. Load a pre-trained language model for text generation.
2. Define a prompt template to guide the LLM with the query and retrieved documents.
3. Create an LLMChain to combine the LLM and prompt template.
4. Generate an answer by providing the query and retrieved documents to the LLMChain.

## Evaluation

The script employs various metrics to evaluate the system's performance, including:

1. BLEU score: Measures the similarity between the generated answer and the reference answer.
2. ROUGE score: Assesses the overlap between the generated and reference answers based on n-grams.
3. Cosine similarity: Measures the semantic similarity between the generated and reference answers.

## Further Exploration
- Experiment with different pre-trained models for retrieval and generation.
- Fine-tune the models on your specific domain or task.
- Explore additional features like answer summarization or ranking.
- Consider using more advanced retrieval techniques like dense retrieval.

Note:

- This is a foundational implementation for demonstration purposes.
- The script employs input truncation to adhere to model limitations.
- Be mindful of memory constraints when scaling to larger datasets.
