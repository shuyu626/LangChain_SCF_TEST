import os
from rich import print as pprint
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel
from langchain.agents import (
    AgentExecutor, create_openai_tools_agent
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from chains.tools_api import fa_list_tool
from prompts.agent_prompt import agent_prompt
from prompts.qa_fewshot_prompt import few_shot_prompt
class SCFQAInput(BaseModel):
    input: str
    role: str
    token: str
    session_id: str
# 載入 .env 檔
load_dotenv()

# 取得 GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("請在 .env 檔中設定 GOOGLE_API_KEY")

# 設定環境變數（如果你後續用的程式需要）
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# llm = ChatOllama(
#     model="gemma3:12b",  # 模型名稱
#     base_url="http://192.168.2.255:11434",  # 你提供的 IP 和 port
# )
# llm = ChatOllama(
#     model="deepseek-r1:14b",  # 模型名稱
#     base_url="http://59.126.229.182:11434",  # 你提供的 IP 和 port
# )

# 嵌入模型
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = Docx2txtLoader("./scf2.0_chatbot.docx")
docs = loader.load()
# 合併成一個大 Document
full_text = "\n".join([doc.page_content for doc in docs])
word_docs = Document(page_content=full_text)
# pprint(word_docs)


text_splitter = RecursiveCharacterTextSplitter(separators=["[一二三四五六七八九十]+、"],
                                                is_separator_regex=True,
                                                chunk_size=300,
                                                chunk_overlap=20)
splits = text_splitter.split_documents([word_docs])
# pprint(splits)

# 建立向量資料庫，將 splits 切好的文件 embed 並儲存
db = FAISS.from_documents(splits, embeddings)

# 儲存向量資料庫到本地目錄
db.save_local(
    folder_path="./scf2.0_db",     # 儲存資料夾
    index_name="scf2.0_test1"      # 儲存檔名前綴
)

# 從本地重新載入向量資料庫
new_db = FAISS.load_local(
    folder_path="./scf2.0_db",
    index_name="scf2.0_test1",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # 若遇 pickle 載入警告需加上
)


str_parser = StrOutputParser()
system_template = """
你是一位專業的【融資平台問答助理】。

【重要指示】
- 請嚴格按照以下格式和標準回應，確保每次回應的一致性
- 先參考回覆的示範，並僅回應文字
- 使用者角色是 {role}，若 {role} 為 'Bank'，代表角色是'銀行'; 'Buyer' 代表'中心廠'; 'Supplier' 代表'供應商'

【回應規則】
1. 角色限制：所有回答僅針對 {role} 角色可執行的操作或權限，不可提及其他角色的流程或操作，若該角色無權限，請回應您並無相關操作權限
2. 語言：使用繁體中文作答
3. 格式：純文字格式，禁止使用任何 Markdown 語法（例如 **、*、`、-、> 等符號）。
4. 結構：
- 步驟說明：使用數字條列 (1. 2. 3.)
- 功能列舉：使用圓點符號 (•)
- 每個步驟或功能點應簡潔明確
5. 內容準確性：回覆內容務必與文件內容一致，若文件中未提及，請明確說「抱歉，您所提的問題可能與 SCF 平台無關，或您目前角色無相關操作權限。」

【標準回應格式範例】
問題類型：操作流程
格式：
請依以下步驟進行：
1. 第一步驟
2. 第二步驟
3. 第三步驟

問題類型：權限/功能查詢
格式：
{role}角色權限包括：
• 功能一
• 功能二
• 功能三


【非相關問題標準回應】
若問題與融資平台無關，請回覆：
「很抱歉，我只能回答有關 融資平台 相關的問題。如果您有任何關於融資平台的問題，歡迎隨時提問！」

[文件內容]
{context}

[問題]
{input}
"""

# qa_prompt = ChatPromptTemplate.from_template(system_template)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    *few_shot_prompt.format_messages(),
    ("human", "使用者角色是 {role}，問題是：{input}")
])

# 建立檢索器
retriever = new_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# retriever 只接受一個字串 query，不接受 dict
def retriever_func(inputs):
    input = inputs["input"]
    return retriever.get_relevant_documents(input)
chain = (
    {"context": retriever_func, "input": RunnablePassthrough(),"role": RunnablePassthrough()}
    | qa_prompt
    | llm
    | str_parser
)

# 工具
def scf_qa_chain_run(query: str) -> str:
    return chain.invoke(query)

def scf_qa_chain_run_stream(input: str, role: str, token: str, session_id: str) -> str:
    response = ""
    
    for chunk in chain.stream({"input": input, "role": role, "token":token, "session_id":session_id}):
        response += chunk
    
    return response

qa_tool = StructuredTool.from_function(
    func=scf_qa_chain_run_stream,
    name="SCF_QA",
    description=(
        "回答使用者關於 SCF 平台操作流程的問題"
        "例如：如何修改密碼？如何審核案件？如何上傳案件？案件流程？如何送出申請？權限有哪些？"
        "請在你判斷與 SCF 操作流程相關時使用這個工具"
    ),
)
tools = [qa_tool,fa_list_tool]

store = {}

# 根據 session_id 建立或取得對應的對話歷史記錄（記憶）
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def print_session_history(session_id: str):
    if session_id in store:
        print(f"📚 Session: {session_id} 記憶內容如下：")
        for i, msg in enumerate(store[session_id].messages):
            print(f"{i+1}. [{msg.type}] {msg.content}")
    else:
        print(f"❌ 查無 session：{session_id}")



# 清除歷史紀錄
def clear_session_history(session_id: str):
    if session_id in store:
        del store[session_id]
        print(f"✅ Session '{session_id}' 的記憶已被清除。")
    else:
        print(f"⚠️ 找不到 session '{session_id}'，無需清除。")

agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose = True)
# 建立有記憶能力的 Agent Chain
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
