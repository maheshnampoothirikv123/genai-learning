HF_API_KEY ="<<Input-Your-HuggingFace-API-Key>>"

from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", huggingfacehub_api_token = HF_API_KEY)

prompt = input("Enter your question:")
response = llm(prompt)

print("=============================")
print("LangChain + Google HuggingFace Example")
print("=============================")

print("Question:" + prompt)
print("Answer:" + response)






