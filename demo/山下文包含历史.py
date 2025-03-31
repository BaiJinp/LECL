from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import SQLChatMessageHistory

# 初始化 LLM
llm = ChatOpenAI(temperature=0.9,
                 model="deepseek-chat",
                 api_key="sk-c6b2a6efae24431892618c047ae1348f",
                 base_url="https://api.deepseek.com")

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    MessagesPlaceholder(variable_name="history"),  # 占位符用于历史消息
    ("user", "{text}"),
    ("assistant", "{text}"),
])

# 输出解析器
parser = StrOutputParser()

# 构建链式调用
chain = prompt | llm | parser

# 获取会话函数
def getSession(sid):
    # 使用 SQLite 数据库存储聊天历史
    return SQLChatMessageHistory(session_id=sid, connection="sqlite:///history.db")

# 将会话历史集成到链中
runnable = RunnableWithMessageHistory(chain, getSession, input_messages_key='text', history_messages_key="history")
# 调用链并传入参数
resp = runnable.invoke({"input_language": "Chinese", "output_language": "Japanese", "text": "小日本"},
                       config={"configurable": {"session_id": "123"}})
# 打印结果
print(resp)
