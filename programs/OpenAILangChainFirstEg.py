OPENAI_API_KEY ="<<Input-Your-OpenAI-API-Key>>"

from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", openai_api_key=OPENAI_API_KEY)

prompt = input("Enter your question:")
response = llm(prompt)

print("=============================")
print("LangChain + OpenAI Example")
print("=============================")

print("Question:" + prompt)
print("Answer:" + response)