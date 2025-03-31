from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.9,
                 model="deepseek-chat",
                 api_key="sk-c6b2a6efae24431892618c047ae1348f",
                 base_url="https://api.deepseek.com")

# 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("user", "{text}"),
    ("assistant", "{text}"),
])

parser = StrOutputParser() # 输出解析器

chain  = prompt | llm | parser
resp = chain.invoke({"input_language": "Chinese", "output_language": "Japanse", "text": "小日本"})
print(resp)