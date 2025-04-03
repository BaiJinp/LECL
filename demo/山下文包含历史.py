import sqlite3

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
# from langchain.memory import SQLChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
# 初始化 LLM
llm = ChatOpenAI(temperature=0.9,
                 model="deepseek-chat",
                 api_key="sk-c6b2a6efae24431892618c047ae1348f",
                 base_url="https://api.deepseek.com")

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你现在是一个地理专家"),
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
    return SQLChatMessageHistory(sid, "sqlite:///history.db")

# 创建 message_history 表
def create_message_history_table(db_path):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS message_history (
        session_id TEXT,
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

# 数据库路径
db_path = "history.db"

# 创建表
create_message_history_table(db_path)

# 将会话历史集成到链中
runnable = RunnableWithMessageHistory(chain, getSession, input_messages_key='text', history_messages_key="history")
# 调用链并传入参数
resp = runnable.invoke({"text": "日本"},
                       config={"configurable": {"session_id": "123"}})

resp1 = runnable.invoke({"text": "这个国家有哪些著名的城市"},
                       config={"configurable": {"session_id": "123"}})
# 打印结果
print(resp)
print(resp1)
