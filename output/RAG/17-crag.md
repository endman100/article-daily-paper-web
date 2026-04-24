---
title: "CRAG：糾錯式 RAG，為檢索結果加裝品質守門員"
description: "傳統 RAG 盲目信任檢索結果，即使取回了完全不相關的文件也照用不誤。CRAG（Corrective RAG）引入評估器對檢索品質評分，根據分數動態選擇：直接使用、完全拋棄改用網路搜尋、或混合兩者，從根本解決 RAG 對低品質檢索的脆弱性。"
date: 2024-01-17
tags: ["RAG", "CRAG", "Corrective RAG", "LangGraph", "Evaluator", "Web Search", "糾錯"]
---

# CRAG：糾錯式 RAG，為檢索結果加裝品質守門員

CRAG（Corrective Retrieval Augmented Generation，糾錯式檢索增強生成）來自 Yan et al. 在 2024 年發表的論文《Corrective Retrieval Augmented Generation》。它的出發點非常務實：現有 RAG 系統在遇到低品質檢索時會「帶著錯誤繼續走」，而 CRAG 的解法是在檢索之後加入一道評估關卡，根據評估結果動態決定下一步行動。這種設計使 CRAG 在高準確性要求的場景中表現遠優於傳統 RAG。

---

## 核心概念

### 傳統 RAG 的根本脆弱性

標準 RAG 管道的問題可以用一個簡單場景說明：

```
問題：「台積電 2nm 製程的良率是多少？」

檢索結果（假設知識庫沒有相關資訊）：
→ 取回了關於「台積電 3nm 製程量產」的文件
→ 文件確實提到了台積電，但完全沒有 2nm 良率的資訊

標準 RAG 的行為：
→ 把這個不相關的文件塞進 prompt
→ LLM 嘗試從不相關的文件中「擠出」答案
→ 產生幻覺（Hallucination）：「根據文件，台積電 2nm 良率為 XX%」（編的）
```

問題的本質是：**標準 RAG 對檢索失敗沒有任何應對機制**，它永遠假設「檢索到的文件就是相關的」。

### CRAG 的三條行動路徑

CRAG 的核心設計是：在檢索完成後，用一個輕量的**評估器（Evaluator）**對每篇文件打分，然後根據分數選擇三條不同的後續路徑：

```
              ┌──────────────────────────────────────────────────┐
              │             Evaluator 評估檢索品質                │
              │  輸入：查詢 Q + 每篇文件 D                       │
              │  輸出：相關性分數（高/中/低）                     │
              └──────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
           高分          中分          低分
         (Correct)    (Ambiguous)   (Incorrect)
              │            │            │
              ▼            ▼            ▼
        知識提煉        混合策略       網路搜尋
   (Knowledge         (KG + Web     (完全拋棄
    Refinement)        Search)       本地文件)
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                    LLM 生成最終答案
```

---

## 運作原理

### 三條路徑詳解

#### 路徑一：Correct（高分） → 知識提煉（Knowledge Refinement）

當評估器認為檢索到的文件高度相關時，並不是直接把整篇文件塞進 prompt，而是進行**知識提煉**：

```
分解（Decompose）→ 篩選（Filter）→ 重組（Recompose）

步驟 1：將文件分解成細粒度的知識片段（段落或句子）
步驟 2：對每個片段評分，篩除與問題無關的噪音
步驟 3：將相關片段重新組合成乾淨的上下文

目的：即使文件整體相關，也可能包含不必要的資訊，
      知識提煉確保送進 LLM 的內容都是有用的
```

#### 路徑二：Incorrect（低分） → 網路搜尋（Web Search）

當本地知識庫的文件完全不相關時，CRAG 會**完全拋棄**本地結果，轉而用**網路搜尋**獲取實時資訊。

這個設計的重要性在於：與其用錯誤的上下文生成幻覺，不如坦誠地去找最新的資訊。

#### 路徑三：Ambiguous（中分） → 混合策略

當評估分數介於高低之間時，CRAG 結合本地文件（知識提煉後）和網路搜尋結果，加權合併後提供給 LLM。

### 評估器設計

CRAG 論文中的評估器是一個輕量的**分類器**（CrossEncoder 或 LLM-as-Evaluator），而不是大型 LLM：

```python
# 評估器輸入：(查詢, 文件) 配對
# 評估器輸出：Correct / Incorrect / Ambiguous

# 論文原始使用 T5-large fine-tuned 的分類器
# 實作時可以用 LLM 模擬（較慢但無需訓練）
```

---

## Python 實作範例

### 使用 LangGraph 實作 CRAG 工作流

```python
from typing import TypedDict, List, Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
import json

# ─────────────────────────────────────
# 步驟 1：定義 CRAG 工作流的狀態
# ─────────────────────────────────────
class CRAGState(TypedDict):
    """CRAG 工作流的完整狀態"""
    query: str                          # 原始查詢
    documents: List[Document]           # 檢索到的文件
    evaluation: str                     # 評估結果：correct / incorrect / ambiguous
    refined_context: str                # 知識提煉後的上下文
    web_results: List[str]              # 網路搜尋結果
    final_answer: str                   # 最終答案
    decision_log: List[str]             # 決策日誌（可解釋性）

# ─────────────────────────────────────
# 步驟 2：建立向量資料庫和 LLM
# ─────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 範例知識庫（實際使用時替換為真實文件）
docs = [
    Document(
        page_content="台積電 3nm 製程（N3）相比 5nm 提升60%的邏輯密度，效能提升15%，功耗降低30%。",
        metadata={"source": "tsmc_tech"}
    ),
    Document(
        page_content="NVIDIA H100 GPU 採用台積電 4nm 製程，搭載 HBM3 記憶體，提供 3.35 TB/s 記憶體頻寬。",
        metadata={"source": "nvidia_h100"}
    ),
    Document(
        page_content="Python 3.12 引入 f-string 的改進，支援在 f-string 中使用反斜線和多行表達式。",
        metadata={"source": "python312"}
    ),
]

vectorstore = Chroma.from_documents(docs, embeddings)

# 網路搜尋工具（需要 Tavily API Key）
# web_search_tool = TavilySearchResults(max_results=3)

# ─────────────────────────────────────
# 步驟 3：定義各個節點（Node）函數
# ─────────────────────────────────────

def retrieve_node(state: CRAGState) -> CRAGState:
    """節點 1：從向量資料庫檢索文件"""
    query = state["query"]
    documents = vectorstore.similarity_search(query, k=3)
    
    log_msg = f"[Retrieve] 從知識庫取回 {len(documents)} 篇文件"
    print(log_msg)
    
    return {
        **state,
        "documents": documents,
        "decision_log": state.get("decision_log", []) + [log_msg]
    }


def evaluate_node(state: CRAGState) -> CRAGState:
    """
    節點 2：評估器（Evaluator）
    對每篇文件進行相關性評估，決定整體品質等級。
    """
    query = state["query"]
    documents = state["documents"]
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個嚴格的文件相關性評估器。
        
評估提供的文件集合是否能回答用戶問題。
根據以下標準給出評估結果：

- "correct"：至少有一篇文件包含足夠回答問題的資訊
- "incorrect"：所有文件都與問題無關，或者根本沒有有用資訊
- "ambiguous"：文件有部分相關資訊，但不完整或不確定

只回答 JSON：{{"evaluation": "correct/incorrect/ambiguous", "reason": "原因", "relevant_docs": [0,1,2] (相關文件的索引列表)}}"""),
        ("human", """問題：{query}
        
文件集合：
{docs_text}

評估結果：""")
    ])
    
    docs_text = "\n\n".join([
        f"文件 {i}: {doc.page_content}"
        for i, doc in enumerate(documents)
    ])
    
    response = llm.invoke(eval_prompt.format_messages(
        query=query,
        docs_text=docs_text
    ))
    
    try:
        result = json.loads(response.content)
        evaluation = result["evaluation"]
        reason = result.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        evaluation = "ambiguous"
        reason = "評估解析失敗，採用保守策略"
    
    log_msg = f"[Evaluate] 結果: {evaluation} | 原因: {reason}"
    print(log_msg)
    
    return {
        **state,
        "evaluation": evaluation,
        "decision_log": state.get("decision_log", []) + [log_msg]
    }


def knowledge_refinement_node(state: CRAGState) -> CRAGState:
    """
    節點 3：知識提煉（Knowledge Refinement）
    適用於 Correct 路徑。
    分解-篩選-重組，去除文件中的噪音。
    """
    query = state["query"]
    documents = state["documents"]
    
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """從以下文件中提取並重組與問題直接相關的知識片段。
去除所有無關的資訊，只保留能回答問題的核心內容。
輸出應該是簡潔、有序的知識摘要，而不是原文重複。"""),
        ("human", """問題：{query}

原始文件：
{docs_text}

提煉後的知識（只包含相關內容）：""")
    ])
    
    docs_text = "\n\n".join([doc.page_content for doc in documents])
    response = llm.invoke(refine_prompt.format_messages(
        query=query,
        docs_text=docs_text
    ))
    
    log_msg = "[Knowledge Refinement] 知識提煉完成"
    print(log_msg)
    print(f"  提煉結果：{response.content[:200]}...")
    
    return {
        **state,
        "refined_context": response.content,
        "decision_log": state.get("decision_log", []) + [log_msg]
    }


def web_search_node(state: CRAGState) -> CRAGState:
    """
    節點 4：網路搜尋
    適用於 Incorrect 路徑。
    完全拋棄本地文件，用網路搜尋獲取最新資訊。
    """
    query = state["query"]
    
    log_msg = "[Web Search] 本地文件不相關，啟動網路搜尋"
    print(log_msg)
    
    # 實際生產中使用 Tavily 或其他搜尋 API
    # web_results = web_search_tool.invoke({"query": query})
    # results_text = [r["content"] for r in web_results]
    
    # 模擬網路搜尋結果（測試用）
    results_text = [
        f"（模擬網路搜尋結果）關於 '{query}' 的最新資訊：這是從網路即時獲取的相關內容..."
    ]
    
    return {
        **state,
        "web_results": results_text,
        "refined_context": "\n".join(results_text),  # 用網路結果作為上下文
        "decision_log": state.get("decision_log", []) + [log_msg]
    }


def hybrid_search_node(state: CRAGState) -> CRAGState:
    """
    節點 5：混合搜尋
    適用於 Ambiguous 路徑。
    結合本地知識提煉 + 網路搜尋。
    """
    query = state["query"]
    documents = state["documents"]
    
    log_msg = "[Hybrid] 結合本地文件 + 網路搜尋"
    print(log_msg)
    
    # 先做知識提煉（取部分本地內容）
    local_context = "\n".join([doc.page_content for doc in documents[:2]])
    
    # 模擬網路搜尋（生產環境替換為真實搜尋）
    web_context = f"（網路補充資訊）{query} 的相關最新資訊..."
    
    # 混合兩個來源
    combined_context = f"【本地知識庫資訊】\n{local_context}\n\n【網路最新資訊】\n{web_context}"
    
    return {
        **state,
        "refined_context": combined_context,
        "decision_log": state.get("decision_log", []) + [log_msg]
    }


def generate_answer_node(state: CRAGState) -> CRAGState:
    """
    節點 6：生成最終答案
    根據精煉後的上下文生成回答。
    """
    query = state["query"]
    context = state.get("refined_context", "")
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個準確、可靠的技術助手。
根據提供的上下文回答問題。
如果上下文資訊不足，請坦誠說明，不要捏造資訊。"""),
        ("human", """上下文資訊：
{context}

問題：{query}

回答：""")
    ])
    
    response = llm.invoke(gen_prompt.format_messages(
        context=context,
        query=query
    ))
    
    log_msg = "[Generate] 最終答案生成完成"
    print(log_msg)
    
    return {
        **state,
        "final_answer": response.content,
        "decision_log": state.get("decision_log", []) + [log_msg]
    }


# ─────────────────────────────────────
# 步驟 4：定義條件路由（根據評估結果分流）
# ─────────────────────────────────────
def route_after_evaluation(state: CRAGState) -> Literal["knowledge_refinement", "web_search", "hybrid_search"]:
    """
    根據評估結果決定下一個節點。
    這是 CRAG 的核心分流邏輯。
    """
    evaluation = state.get("evaluation", "ambiguous")
    
    if evaluation == "correct":
        print("→ 路由到：知識提煉（Correct 路徑）")
        return "knowledge_refinement"
    elif evaluation == "incorrect":
        print("→ 路由到：網路搜尋（Incorrect 路徑）")
        return "web_search"
    else:  # ambiguous
        print("→ 路由到：混合搜尋（Ambiguous 路徑）")
        return "hybrid_search"


# ─────────────────────────────────────
# 步驟 5：建立 LangGraph 工作流
# ─────────────────────────────────────
workflow = StateGraph(CRAGState)

# 加入所有節點
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("knowledge_refinement", knowledge_refinement_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("hybrid_search", hybrid_search_node)
workflow.add_node("generate", generate_answer_node)

# 設定起始節點
workflow.set_entry_point("retrieve")

# 設定邊（流程連接）
workflow.add_edge("retrieve", "evaluate")

# 條件路由：evaluate → 三條路徑之一
workflow.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "knowledge_refinement": "knowledge_refinement",
        "web_search": "web_search",
        "hybrid_search": "hybrid_search"
    }
)

# 三條路徑都最終匯聚到 generate
workflow.add_edge("knowledge_refinement", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("hybrid_search", "generate")
workflow.add_edge("generate", END)

# 編譯工作流
crag_app = workflow.compile()


# ─────────────────────────────────────
# 步驟 6：執行測試
# ─────────────────────────────────────
def run_crag(question: str) -> dict:
    """執行 CRAG 工作流"""
    print(f"\n{'='*60}")
    print(f"問題：{question}")
    print('='*60)
    
    initial_state = CRAGState(
        query=question,
        documents=[],
        evaluation="",
        refined_context="",
        web_results=[],
        final_answer="",
        decision_log=[]
    )
    
    result = crag_app.invoke(initial_state)
    
    print(f"\n{'─'*40}")
    print("決策日誌：")
    for log in result["decision_log"]:
        print(f"  {log}")
    
    print(f"\n最終答案：")
    print(result["final_answer"])
    
    return result

# 測試 1：知識庫有答案（Correct 路徑）
result1 = run_crag("台積電 3nm 製程的效能提升了多少？")

# 測試 2：知識庫沒有答案（Incorrect 路徑）
result2 = run_crag("最新一代 iPhone 的電池容量是多少？")

# 測試 3：部分相關（Ambiguous 路徑）
result3 = run_crag("NVIDIA 的 GPU 使用什麼記憶體技術，最新的良率如何？")
```

---

## 優缺點分析

### 優點

**1. 主動防禦低品質檢索**
CRAG 不假設檢索永遠成功，而是把「檢索失敗」作為一種正常狀態納入設計，大幅降低幻覺風險。

**2. 動態回退到最新資訊**
當本地知識庫過時或不包含相關資訊時，CRAG 能自動回退到網路搜尋，確保答案的時效性。

**3. 無需微調模型**
與 Self-RAG 不同，CRAG 是一個架構層面的設計，不需要對任何 LLM 進行微調，可以即插即用到現有 RAG 系統。

**4. 知識提煉降低噪音**
即使檢索成功，知識提煉步驟也會過濾文件中的無關內容，進一步提升生成質量。

**5. 可解釋的決策路徑**
工作流的每個步驟都有清晰的決策邏輯，便於除錯和監控。

### 缺點

**1. 增加系統複雜度**
CRAG 引入了評估器、知識提煉、網路搜尋等多個額外組件，部署和維護成本更高。

**2. 評估器的準確性是瓶頸**
如果評估器評分不準（把相關文件誤判為不相關），會導致不必要的網路搜尋，既浪費資源又可能引入不一致的資訊。

**3. 額外的延遲**
評估器調用 + 條件性的知識提煉/網路搜尋，至少增加 1-3 秒的延遲。

**4. 網路搜尋的一致性問題**
當切換到網路搜尋時，搜尋結果的品質、格式、來源可信度都會有所不同，需要額外的品質控制。

---

## 適用場景

### 最適合 CRAG 的場景

**高準確性要求的問答系統**
醫療諮詢、法律顧問、金融分析等場景，答案必須準確、有來源依據，CRAG 的多重品質把關機制是理想選擇。

**知識庫覆蓋不完整的系統**
當本地知識庫不可能涵蓋所有可能被問到的問題時，CRAG 的網路搜尋回退機制能優雅地處理知識空白。

**需要時效性資訊的應用**
新聞摘要、市場分析、技術更新追蹤等場景，問題可能超出本地知識庫的更新時間範圍，網路搜尋回退尤為重要。

**企業客服機器人**
企業客服需要同時處理產品手冊（本地知識庫）和即時市場資訊（網路搜尋），CRAG 的混合策略完美契合。

### 不適合的場景

- 延遲極敏感的即時應用
- 完全離線、不能訪問網路的部署環境
- 知識庫覆蓋非常完整、很少出現「不相關」結果的場景

---

## 與其他方法的比較

### CRAG vs Self-RAG

這兩個方法都試圖解決 RAG 對低品質檢索的脆弱性，但設計哲學完全不同：

| 特性 | CRAG | Self-RAG |
|------|------|---------|
| 糾錯位置 | 外部評估器 | 模型內部 |
| 需要微調 | 否 | 是 |
| 決策透明度 | 高（明確的路由邏輯）| 中（Token 可見但非顯式邏輯）|
| 回退機制 | 網路搜尋 | 無（依賴模型自身知識）|
| 插拔性 | 高（可加入任何 RAG 系統）| 低（需要特定 fine-tuned 模型）|
| 工程複雜度 | 中 | 高（訓練）/ 低（使用現有模型）|

**結論**：CRAG 更適合工程師快速部署，因為不需要訓練；Self-RAG 需要微調訓練，但決策是模型內化的，推論時不依賴外部邏輯。

### CRAG vs 傳統 RAG + Reranker

傳統方案是在 RAG 管道中加入 Reranker（重排序器）來改善檢索品質：

- **Reranker**：改善文件排序，選出最相關的前 K 篇，但仍然使用本地文件
- **CRAG**：不僅評估相關性，還能完全切換到網路搜尋

CRAG 的優勢在於：當本地知識庫根本沒有答案時，Reranker 無論怎麼排序都無法改善結果，而 CRAG 能識別這種情況並採取行動。

### 三種方法的組合使用

實際生產系統中，這些方法不是互斥的：

```
查詢
  │
  ▼
HyDE 查詢擴展（改善搜尋向量）
  │
  ▼
Hierarchical Indices 搜尋（精確定位）
  │
  ▼
Reranker 重排序（選出最相關）
  │
  ▼
CRAG 評估器（品質把關）
  │
  ├── Correct → 知識提煉 → LLM 生成
  ├── Ambiguous → 混合策略 → LLM 生成
  └── Incorrect → 網路搜尋 → LLM 生成
```

這個完整管道組合了多種技術，能處理各種邊緣情況，是生產級 RAG 系統的參考架構。

---

## 小結

CRAG 的核心貢獻是將 RAG 系統從「樂觀主義」（假設檢索永遠成功）轉變為「務實主義」（承認檢索可能失敗並設計應對方案）。

透過評估器 + 三條行動路徑的設計，CRAG 能在不同的檢索品質情況下採取最優策略：
- 高品質檢索 → 知識提煉後直接使用
- 低品質檢索 → 完全切換到網路搜尋
- 中間狀態 → 混合兩者

這個框架的實際部署已被 LangGraph 等工具很好地支援，工程師可以用相對簡潔的代碼實現完整的 CRAG 工作流。在高準確性要求的應用場景（醫療、法律、金融）中，CRAG 的多重品質把關機制往往能帶來顯著的可信度提升，是現代 RAG 工程實踐中不可忽視的重要技術。
