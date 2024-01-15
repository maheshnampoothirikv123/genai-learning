from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings.openai import OpenAIEmbeddings

file = '<<Your CSV File path>>'
loader = CSVLoader(file_path=file, encoding='utf8')
documents = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = DocArrayInMemorySearch.from_documents(
    documents,
    embeddings
)

template = ("Use the following pieces of context to answer the question. Provide precise summary at the end. If you "
            "don't know the answer, say that you don't know, don't try to create an answer. {context} Question: {"
            "question} Answer with description: and Summary:")
PROMPT_TEMPLATE = PromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
)

question = "<<Your Question goes here>>"
response = chain({"query": question})

print(response["query"])
print(response["result"])


