from rich import print as pprint
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import requests

def authorized_get(url: str, token: str, params: dict = None):
    headers = {"Authorization": f"Token {token}"}
    response = requests.get(url, headers=headers, params=params)
    return response

class FAListQueryInput(BaseModel):
    token: str = Field(..., description="使用者身份驗證的 Bearer Token")
    caseNum: str | None = Field(None, description="案件編號")
    invoiceId: str | None = Field(None, description="發票編號")
    caseStatus: str | None = Field(None, description="案件狀態")
    creationtimestart: str | None = Field(None, description="建立時間起 (yyyy-mm-dd)")
    creationtimeend: str | None = Field(None, description="建立時間訖 (yyyy-mm-dd)")

def get_fa_list_from_backend(
    token: str,
    caseNum: str = None,
    invoiceId: str = None,
    caseStatus: str = None,
    creationtimestart: str = None,
    creationtimeend: str = None,
):
    try:
        params = {
            k: v for k, v in {
            "caseNum": caseNum,
            "invoiceId": invoiceId,
            "caseStatus": caseStatus,
            "creationtimestart": creationtimestart,
            "creationtimeend": creationtimeend
        }.items() if v is not None
    }

        url = "http://192.168.0.104:8000/scf-api/transactions/fa/list"
        response = authorized_get(url, token=token, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("payload") and data["payload"].get("data"):
                real_data = data["payload"]["data"]
                if not real_data:
                    return "查無資料"
                return real_data  # 直接回傳案件列表
            else:
                return "API 回傳格式錯誤或無資料"
        else:
            return f"API 回傳錯誤，狀態碼：{response.status_code}"

    except Exception as e:
        return f"API 呼叫失敗，錯誤訊息：{str(e)}"
fa_list_tool = StructuredTool.from_function(
    name="get_fa_list",
    description=(
        "用於查詢案件清單，根據使用者提供的條件：案件編號(caseNum)、"
        "發票號碼(invoiceId)、案件狀態(caseStatus)、以及建立日期區間(creationtimestart, creationtimeend)，"
        "時間格式為 yyyy-mm-dd，或使用者會提到今天、昨天、這個月、近半年、近三個月等時間字眼"
        "使用者可能會詢問「近三個月案件有幾筆」、「所有案件」、「查詢某狀態案件」等問題，"
        "必須帶上授權 token 才能查詢。"
        "若問題包含這些條件，請使用此工具查詢。"
    ),
    func=get_fa_list_from_backend,
    args_schema=FAListQueryInput,
)