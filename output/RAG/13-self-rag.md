---
title: "Self-RAG：讓 LLM 自己決定何時檢索、如何評估"
description: "Self-RAG 透過訓練 LLM 插入特殊 Reflection Token，讓模型在推論時動態決定是否需要檢索、檢索結果是否相關、以及答案是否可信，從根本解決傳統 RAG 無腦檢索的問題。"
date: 2024-01-13
tags: ["RAG", "Self-RAG", "Reflection Token", "LLM", "自適應檢索"]
---

# Self-RAG：讓 LLM 自己決定何時檢索、如何評估

Self-RAG（Self-Reflective Retrieval-Augmented Generation）來自 Asai et al. 在 2023 年發表的論文《Self-RAG: Learning to Retrieve, Generate, and Critique》。它的核心突破在於：不是由外部系統控制檢索邏輯，而是訓練 LLM 本身具備「反思能力」，讓模型在生成過程中自行判斷是否需要檢索、檢索結果是否有用、以及最終答案是否可信。相較於傳統 RAG 的被動接受，Self-RAG 讓模型成為主動的知識消費者。

---

## 核心概念

### 傳統 RAG 的根本問題

傳統 RAG 的邏輯非常簡單粗暴：

```
用戶問問題 → 無論如何都先檢索 → 把檢索結果塞進 prompt → 生成答案
```

這個流程存在兩個嚴重缺陷：

**1. 不必要的檢索（Over-retrieval）**
對於「2 + 2 等於多少？」這類問題，根本不需要任何外部文件。但傳統 RAG 會強制檢索，不僅浪費運算資源，還可能引入噪音。

**2. 缺乏批判能力（Uncritical Acceptance）**
即使檢索回來的文件根本不相關，傳統 RAG 也會「照單全收」，用錯誤的上下文生成答案，產生幻覺（hallucination）。

### Self-RAG 的解法

Self-RAG 的核心思想是：**讓 LLM 學會插入特殊的 Reflection Token**，在生成過程中對自己的行為進行「反思」。

這些 Reflection Token 是在訓練階段注入的特殊符號，不是由外部邏輯控制，而是模型本身學習到什麼時候應該輸出哪個 token。

---

## 運作原理

### 四個 Reflection Token

Self-RAG 定義了四種關鍵的 Reflection Token：

| Token | 意義 | 可能的值 |
|-------|------|---------|
| `[Retrieve]` | 模型判斷是否需要檢索 | `yes` / `no` / `continue` |
| `[IsREL]` | 檢索到的文件是否與問題相關 | `relevant` / `irrelevant` |
| `[IsSUP]` | 生成的回答是否被文件支持 | `fully supported` / `partially supported` / `not supported` |
| `[IsUSE]` | 整體回答對用戶是否有用 | `1` ~ `5`（評分） |

### 推論流程

```
用戶輸入問題
      │
      ▼
┌─────────────────────────────┐
│  LLM 生成 [Retrieve] token  │
│  判斷是否需要檢索            │
└─────────────────────────────┘
      │
   ┌──┴──┐
[yes]   [no]
  │       │
  ▼       ▼
呼叫    直接生成答案
檢索器  （跳過檢索）
  │
  ▼
取回 N 個候選文件
  │
  ▼
┌─────────────────────────────────┐
│  對每個文件，LLM 生成：          │
│  [IsREL] → 文件相關性評估        │
│  生成答案片段                    │
│  [IsSUP] → 答案是否被文件支持    │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  用 [IsUSE] 對所有候選答案  │
│  評分，選出最佳答案          │
└─────────────────────────────┘
  │
  ▼
輸出最終答案
```

### 訓練方式

Self-RAG 並不是用 prompt engineering 實現的，而是**真正微調（fine-tune）**了語言模型。訓練資料包含了人工標注的 Reflection Token，讓模型學習在適當位置插入這些反思標記。

這意味著 Self-RAG 的能力是模型內建的，不依賴外部邏輯，推論速度更快，決策更一致。

---

## Python 實作範例

以下用 LangChain + GPT-4 模擬 Self-RAG 的核心邏輯（概念實作，非原始論文的 fine-tuned 版本）：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple
import json

# 初始化模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# ─────────────────────────────────────
# 步驟 1：建立向量資料庫（示範用）
# ─────────────────────────────────────
docs = [
    Document(page_content="台積電（TSMC）於1987年由張忠謀在台灣新竹創立，是全球最大的晶圓代工廠。"),
    Document(page_content="TSMC 的 3nm 製程技術於2022年正式量產，主要供應 Apple 和 NVIDIA。"),
    Document(page_content="Python 是一種高階程式語言，以簡潔的語法著稱，廣泛用於資料科學和 AI 開發。"),
    Document(page_content="大型語言模型（LLM）透過 Transformer 架構訓練，能理解和生成自然語言。"),
]

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ─────────────────────────────────────
# 步驟 2：實作各個 Reflection Token 的判斷邏輯
# ─────────────────────────────────────

def decide_retrieve(query: str) -> bool:
    """
    模擬 [Retrieve] token 的決策邏輯。
    判斷問題是否需要外部知識來回答。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個判斷是否需要檢索外部資料的助手。
        
分析以下問題，判斷回答時是否需要檢索外部文件。
以下情況不需要檢索：
- 數學計算（如 2+2=4）
- 通用常識（如太陽是恆星）
- 簡單定義

以下情況需要檢索：
- 特定事實、數據、日期
- 專業領域的具體資訊
- 最新發展或特定實體的資訊

只回答 JSON 格式：{{"retrieve": true/false, "reason": "原因"}}"""),
        ("human", "問題：{query}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=query))
    result = json.loads(response.content)
    print(f"[Retrieve] = {result['retrieve']} | 原因：{result['reason']}")
    return result['retrieve']


def evaluate_relevance(query: str, doc: Document) -> str:
    """
    模擬 [IsREL] token。
    評估檢索到的文件是否與問題相關。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """評估以下文件是否與問題相關。
只回答 JSON：{{"relevant": true/false, "score": 0.0-1.0}}"""),
        ("human", "問題：{query}\n\n文件內容：{doc_content}")
    ])
    
    response = llm.invoke(prompt.format_messages(
        query=query,
        doc_content=doc.page_content
    ))
    result = json.loads(response.content)
    status = "relevant" if result['relevant'] else "irrelevant"
    return status


def generate_with_doc(query: str, doc: Document) -> str:
    """
    根據單一文件生成答案片段。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "根據提供的文件回答問題。只使用文件中的資訊，不要添加額外知識。"),
        ("human", "問題：{query}\n\n文件：{doc_content}\n\n回答：")
    ])
    
    response = llm.invoke(prompt.format_messages(
        query=query,
        doc_content=doc.page_content
    ))
    return response.content


def evaluate_support(answer: str, doc: Document) -> str:
    """
    模擬 [IsSUP] token。
    評估答案是否被文件內容支持。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """評估答案是否有文件支持。
只回答其中之一：fully_supported / partially_supported / not_supported"""),
        ("human", "文件：{doc_content}\n\n答案：{answer}")
    ])
    
    response = llm.invoke(prompt.format_messages(
        doc_content=doc.page_content,
        answer=answer
    ))
    return response.content.strip()


def evaluate_usefulness(query: str, answer: str) -> int:
    """
    模擬 [IsUSE] token。
    對整體答案的有用程度評分（1-5）。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """對答案的有用程度評分（1=完全無用，5=非常有用）。
只回答數字 1-5。"""),
        ("human", "問題：{query}\n\n答案：{answer}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=query, answer=answer))
    try:
        return int(response.content.strip())
    except ValueError:
        return 3  # 預設中等分數


# ─────────────────────────────────────
# 步驟 3：Self-RAG 主流程
# ─────────────────────────────────────

def self_rag(query: str) -> dict:
    """
    Self-RAG 完整推論流程。
    返回最終答案及決策過程。
    """
    print(f"\n{'='*50}")
    print(f"問題：{query}")
    print('='*50)
    
    # ── [Retrieve]：決定是否需要檢索 ──
    needs_retrieval = decide_retrieve(query)
    
    if not needs_retrieval:
        # 直接生成，不檢索
        response = llm.invoke(f"請回答以下問題：{query}")
        return {
            "answer": response.content,
            "retrieved": False,
            "usefulness": evaluate_usefulness(query, response.content)
        }
    
    # ── 執行檢索 ──
    retrieved_docs = retriever.invoke(query)
    print(f"\n檢索到 {len(retrieved_docs)} 篇文件")
    
    # ── 對每篇文件進行評估和生成 ──
    candidates: List[Tuple[str, str, int]] = []  # (answer, support_level, usefulness)
    
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- 文件 {i+1} ---")
        print(f"內容：{doc.page_content[:80]}...")
        
        # [IsREL]：評估文件相關性
        relevance = evaluate_relevance(query, doc)
        print(f"[IsREL] = {relevance}")
        
        if relevance == "irrelevant":
            print("→ 跳過不相關文件")
            continue
        
        # 根據相關文件生成答案
        answer = generate_with_doc(query, doc)
        
        # [IsSUP]：評估答案支持程度
        support = evaluate_support(answer, doc)
        print(f"[IsSUP] = {support}")
        
        # [IsUSE]：評估有用程度
        usefulness = evaluate_usefulness(query, answer)
        print(f"[IsUSE] = {usefulness}/5")
        
        candidates.append((answer, support, usefulness))
    
    if not candidates:
        # 所有文件都不相關，回退到直接生成
        response = llm.invoke(f"請回答以下問題（沒有相關資料）：{query}")
        return {
            "answer": response.content,
            "retrieved": True,
            "note": "所有檢索結果不相關，使用模型本身知識"
        }
    
    # ── 選出最佳答案（優先選 fully_supported 且 usefulness 最高的）──
    def score_candidate(candidate):
        _, support, usefulness = candidate
        support_score = {"fully_supported": 3, "partially_supported": 1, "not_supported": -1}
        return support_score.get(support, 0) + usefulness
    
    best_answer, best_support, best_usefulness = max(candidates, key=score_candidate)
    
    print(f"\n{'='*50}")
    print(f"最終答案（支持程度：{best_support}，有用性：{best_usefulness}/5）：")
    print(best_answer)
    
    return {
        "answer": best_answer,
        "retrieved": True,
        "support_level": best_support,
        "usefulness": best_usefulness,
        "candidates_count": len(candidates)
    }


# ─────────────────────────────────────
# 測試
# ─────────────────────────────────────
if __name__ == "__main__":
    # 測試 1：需要檢索的問題
    result1 = self_rag("台積電是什麼時候成立的？")
    
    # 測試 2：不需要檢索的問題
    result2 = self_rag("2 加 2 等於多少？")
    
    # 測試 3：專業問題
    result3 = self_rag("TSMC 的最新製程技術是什麼？")
```

---

## 優缺點分析

### 優點

**1. 按需檢索（On-demand Retrieval）**
不是每次都強制檢索，對於簡單問題直接回答，降低延遲和成本。

**2. 主動品質控制**
透過 [IsREL]、[IsSUP]、[IsUSE] 三重把關，大幅降低因低品質檢索導致的幻覺。

**3. 可解釋性更強**
Reflection Token 提供了決策的透明度，可以追蹤模型「為什麼」做某個選擇。

**4. 實驗效果顯著**
在 ASQA（開放域問答）、TriviaQA（知識問答）、ARC-Challenge（推理）等基準上，Self-RAG 均優於傳統 RAG 和沒有 RAG 的基線模型。

### 缺點

**1. 需要微調**
Self-RAG 的核心能力來自微調，不能直接用 prompt engineering 完美複製。使用完整版需要訓練資源。

**2. 推論速度較慢**
多次呼叫 LLM 進行評估（[IsREL]、[IsSUP]、[IsUSE]）會增加延遲，對即時應用是挑戰。

**3. 評估標準的主觀性**
[IsUSE] 的 1-5 評分本身帶有主觀性，模型的自我評估不一定完全準確。

**4. 訓練資料依賴**
Reflection Token 的品質取決於訓練資料的標注質量，標注錯誤會導致模型學到錯誤的反思模式。

---

## 適用場景

### 最適合 Self-RAG 的場景

**高可信度問答系統**
醫療諮詢、法律顧問、金融分析等需要「每一句話都要有依據」的場景，Self-RAG 的三重評估機制能大幅提升可信度。

**混合知識需求的應用**
系統需要同時處理「需要查資料」和「不需要查資料」的問題，Self-RAG 的自適應檢索比固定策略更高效。

**資源受限的生產環境**
相比傳統 RAG 每次都檢索，Self-RAG 的按需策略可以顯著降低向量資料庫的查詢頻率。

### 不適合的場景

- **實時性要求極高**：多輪評估增加延遲
- **無法微調模型**：需要真正 fine-tune 才能發揮最大效果
- **知識庫更新頻繁**：Self-RAG 的評估能力是針對訓練時的知識分佈學習的

---

## 與其他方法的比較

| 特性 | 傳統 RAG | Self-RAG | CRAG | 
|------|---------|---------|------|
| 檢索時機 | 每次都檢索 | 按需檢索 | 每次都檢索但評估 |
| 結果評估 | 無 | 模型內建 | 外部評估器 |
| 需要微調 | 否 | 是 | 否 |
| 決策位置 | 外部系統 | 模型內部 | 外部系統 |
| 可解釋性 | 低 | 高（Token 可見）| 中 |
| 延遲 | 中 | 高 | 高 |

**vs. CRAG（糾錯式 RAG）**
CRAG 使用外部評估器在檢索後評估結果，邏輯類似但架構不同。Self-RAG 的優勢在於決策是模型內化的，不依賴外部評估器的設計；CRAG 的優勢在於不需要微調，可插拔性更好。

**vs. 傳統 Agentic RAG**
Agentic RAG 透過 agent 框架控制何時檢索，Self-RAG 則是讓 LLM 自身具備這個能力。前者更靈活、更易定制；後者更快速、更一致。

---

## 小結

Self-RAG 代表了 RAG 技術從「管道工程」走向「模型能力」的重要轉變。它的核心貢獻不只是一個新的架構，而是提出了一個新的訓練範式：讓模型本身學會「元認知」——知道自己何時需要幫助，以及如何評估幫助的品質。

四個 Reflection Token（[Retrieve]、[IsREL]、[IsSUP]、[IsUSE]）構成了一套完整的自我評估體系，使得 Self-RAG 在多個基準上優於傳統方法。

對於需要在生產環境中部署高可信度問答系統的工程師，Self-RAG 的概念值得深入研究。即使沒有資源完整微調，理解其設計哲學也能幫助我們在 prompt engineering 和 agentic 框架中設計更好的評估邏輯。
