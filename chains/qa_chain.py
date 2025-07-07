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
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.tools.simple import Tool
from langchain.agents import (
    AgentExecutor, create_openai_tools_agent
)
from langchain.tools import StructuredTool

from langchain_core.prompts.chat import MessagesPlaceholder
from pydantic import BaseModel
from chains.tools_api import fa_list_tool
class SCFQAInput(BaseModel):
    question: str
    role: str
    token: str
# 載入 .env 檔
load_dotenv()

# 取得 GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("請在 .env 檔中設定 GOOGLE_API_KEY")

# 設定環境變數（如果你後續用的程式需要）
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


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
template = """
你是一位專業的【融資平台問答助理】。


使用者角色是{role}

請根據下列文件內容，先判斷使用者角色，並結合自身判斷，以條理清晰、正式的方式回應用戶問題：
- 使用繁體中文作答
- 如果有提供角色，所有回答僅針對該角色可執行的操作或權限，不可提及其他角色的流程或操作
- 請務必以純文字格式回覆，禁止使用任何 Markdown 語法（例如 **、*、`、-、> 等符號）。
- 若內容為步驟，請使用條列式，盡量簡潔明確;若內容為資訊列舉（功能/權限/可執行項），則使用圓點（•）
- 回覆內容務必與下列文件內容一致，若文件中未提及，請明確說「文件中未提及」
- 若問題與融資平台無關，請回覆：
「很抱歉，我只能回答有關 融資平台 相關的問題。如果您有任何關於融資平台的問題，歡迎隨時提問！」

[文件內容]
{context}

[問題]
{question}
"""
qa_prompt = ChatPromptTemplate.from_template(template)


# 建立檢索器
retriever = new_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# retriever 只接受一個字串 query，不接受 dict
def retriever_func(inputs):
    question = inputs["question"]
    return retriever.get_relevant_documents(question)
chain = (
    {"context": retriever_func, "question": RunnablePassthrough(),"role": RunnablePassthrough()}
    | qa_prompt
    | llm
    | str_parser
)
# response = chain.invoke("哪裡修改密碼")
# print(response)

# 串流
# for chunk in chain.stream("有哪些權限"):
#     print(chunk, end="", flush=True)


# 工具
def scf_qa_chain_run(query: str) -> str:
  return chain.invoke(query)

def scf_qa_chain_run_stream(question: str, role: str, token: str) -> str:
  print(input)
  response = ""
  for chunk in chain.stream({"question": question, "role": role, "token":token}):
    response += chunk
  return response

qa_tool = StructuredTool.from_function(
    func=scf_qa_chain_run_stream,
    name="SCF_QA",
    description=(
      "回答使用者關於 SCF 平台操作流程的問題，例如：如何修改密碼？"
      "如何送出申請？權限有哪些？"
      "請在你判斷與 SCF 操作流程相關時使用這個工具"
  ),
)
tools = [qa_tool,fa_list_tool]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 SCF 融資平台助理，請根據使用者角色及問題選擇正確的工具回覆問題"),
    ("system", "你必須使用 token 權杖來呼叫任何需要授權的工具，token 是：{token}"),
    ("system", "使用者可能會透過時間的模糊查詢來查詢資料，如果輸入『今天』、『昨天』、『這個月』、『近三個月』、『近半年』等詞，請你依照系統日期轉換成 yyyy-mm-dd 格式的具體日期區間或日期，例如「近半年」轉成「2025-01-07 ~ 2025-07-07」。沒有提及時間點，就是顯示所有時間點"),
    ("system", "如果使用者提到「所有案件」、「全部案件」、「所有狀態的案件」等字眼，請預設查詢所有狀態的案件，不加案件狀態篩選條件。"),
    ("system", "若使用者想查詢案件，則列點提供符合條件的案件編號，及顯示共幾筆資料即可"),
    ("system", 
      "案件狀態值的表示如下：\n"
      "- 案件待上傳：000\n"
      "- 案件待提交：001\n"
      "- 中心廠審核：010\n"
      "- 中心廠審核通過：020\n"
      "- 銀行審核中：110\n"
      "- 銀行審核通過：120\n"
      "- 融資退件：140\n"
      "- 案件更新失敗：902、912、914"
    ),
    ("human", "使用者角色是：{role}，問題是：{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose = True)

# response = agent_executor.invoke({
#     "input": "怎麼修改密碼"
# })
# print(response['output'])
