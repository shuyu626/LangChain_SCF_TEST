# backend/main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from chains.qa_chain import agent_executor  # 匯入你寫好的 function

# 初始化一個 FastAPI 應用實例
app = FastAPI()

# 設定 CORS 中介軟體，允許前端從其他網域呼叫此 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有網域來源進行請求（生產環境建議改成具體網址）
    allow_credentials=True, # 允許瀏覽器攜帶 cookies 或 headers 資訊
    allow_methods=["*"], # 允許所有 HTTP 方法（GET, POST, PUT, DELETE 等）
    allow_headers=["*"], # 允許所有類型的自定義標頭
)

# 定義用戶端送出請求的資料結構，這裡只需要一個「question」欄位
class QueryRequest(BaseModel):
    question: str
    role:str
    token: str

# 定義回傳給用戶端的資料結構，回應是一個字典格式的「answer」
class QueryResponse(BaseModel):
    answer: dict

# 定義一個 POST API 路由 "/ask"，接收用戶的問題並回應答案
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    # 呼叫你自定義的 agent_executor，將問題送進去處理
    response = agent_executor.invoke({"question": request.question,"role":request.role,"token":request.token})
    print(request)
    # print(response)
    
    # 將處理好的結果包裝成 QueryResponse 回傳
    return QueryResponse(answer=response)