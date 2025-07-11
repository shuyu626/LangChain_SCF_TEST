from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)
examples = [
  {
    "input": "上週新增哪些案件？",
    "output": "以下為符合條件的案件：\n• 案件編號：0001889011202507090002，案件 ID：1652ed58-1b73-430e-be91-4c0ee982c42b\n 總筆數：1 筆"
  },
  {
    "input": "哪些案件還沒審核",
    "output": "以下為符合條件的案件：\n• 案件編號：0001889011202507090002，案件 ID：1652ed58-1b73-430e-be91-4c0ee982c42b\n 總筆數：1 筆"
  },
  {
    "input": "0001889011202507090005？",
    "output": "以下為搜尋結果：\n•例如：案件編號：0001889011202507090002，案件 ID：1652ed58-1b73-430e-be91-4c0ee982c42b，案件狀態：融資退件\n 融資金額：1000000 TWD\n\n  "
  },
  {
    "input": "如何上傳案件？",
    "output": "銀行並不具有上傳案件的操作權限。"
  },
]


example_prompt = ChatPromptTemplate.from_messages(
[ ('human', '{input}'),
  ('ai', '{output}')]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)