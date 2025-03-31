from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.9,
                 model="deepseek-chat",
                 api_key="sk-c6b2a6efae24431892618c047ae1348f",
                 base_url="https://api.deepseek.com")


