import langchain
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain.vectorstores import Chroma
from langchain_experimental.sql import SQLDatabaseChain

# "True" prints all the logs to show internal states. "False" makes Off.
langchain.verbose = False

config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0, 'context_length': 10000}

# CodeLlama-7B is memory optimized model to run in CPU/local laptop.
llm = CTransformers(model="TheBloke/CodeLlama-7B-Instruct-GGUF",
                    model_file="codellama-7b-instruct.Q4_K_M.gguf", config=config, verbose=False)

db = SQLDatabase.from_uri('mysql+pymysql://<<username>>:<<password>>@localhost:3306/<<schema>>',
                          # Include only required tables to reduce tokens. Comment to consider all.
                          include_tables=['<<Table#1>>', '<<Table#2>>', '<<Table#3>>'])
"""
# Example of "zero shot" training using SQL DB Chain from LangChain.
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False, return_sql=False, use_query_checker=True)
question = input("Enter your question:")
result = db_chain.run(question)
print(result)
"""

examples = [
    {
        "input": "<<Example question#1 which a user can possibly ask>>",
        "sql": "<<SQL query which can retrieve the answer for the above question#1>>",
        "result": "<<SQL Result set representation>>",
        "answer": "<<Final Answer expected from LLM App>>>",
    },
    {
        "input": "<<Example question#2 which a user can possibly ask>>",
        "sql": "<<SQL query which can retrieve the answer for the above question#2>>",
        "result": "<<SQL Result set representation>>",
        "answer": "<<Final Answer expected from LLM App>>>",
    }
]

EXAMPLE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "sql", "result", "answer", ],
    template="\nQuestion: {input}\nSQLQuery: {sql}\nSQLResult: {result}\nAnswer: {answer}",
)

# Create embeddings to measure semantic similarity.
embeddings = HuggingFaceEmbeddings()
# Create concatenate string to be vectorized.
examples_series = [" ".join(example.values()) for example in examples]
# store the embeddings to Vector DB.
vectorstore = Chroma.from_texts(examples_series, embeddings, metadatas=examples)
# semantic selection from vector embeddings.
example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=1)

# Another way of implementing the semantic selection from vector embeddings in one line.
# example_selector = SemanticSimilarityExampleSelector.from_examples(examples, embeddings, Chroma, k=1)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=EXAMPLE_PROMPT_TEMPLATE,
    prefix=_mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"],
)

# Example of "few shot" training using SQL DB Chain from LangChain.
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=few_shot_prompt, use_query_checker=True, verbose=False,
                                     return_sql=False)
question = input("Enter your question:")
result = db_chain.run(question)
print(result)


