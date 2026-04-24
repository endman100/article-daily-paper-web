---
title: "Query Transformation：用查詢改寫突破語義檢索的天花板"
description: "使用者查詢往往模糊、不完整或語義單一，導致 RAG 遺漏大量相關文件。Query Rewriting、Multi-Query、HyDE、Step-Back Prompting 四種技術從不同角度補齊這個缺陷。"
date: "2025-01-15"
tags: ["RAG", "Query Transformation", "Multi-Query", "HyDE", "Step-Back Prompting", "RRF", "LangChain", "向量搜尋"]
---

# Query Transformation：用查詢改寫突破語義檢索的天花板

使用者輸入「怎麼讓模型不要亂說話」，系統找不到任何結果——但知識庫裡有完整的「幻覺抑制（Hallucination Mitigation）」文件。問題不在文件，在查詢。使用者的自然語言和文件的技術語言之間有一道鴻溝，Query Transformation 就是架在這道鴻溝上的橋。

## 核心概念

**Query Transformation（查詢改寫）** 在查詢進入向量搜尋之前，用 LLM 對查詢本身進行改造。改造可以是一對一的改寫（更精確），也可以是一對多的擴展（更多角度），還可以是生成假設答案來替代查詢（語義空間跳躍）。

四種主要技術的定位：

```
使用者原始查詢
        │
        ▼
┌───────────────────────────────────┐
│        Query Transformation       │
│                                   │
│  Query Rewriting  → 1 個精確查詢  │
│  Multi-Query      → 3-5 個子查詢  │
│  HyDE             → 1 個假設答案  │
│  Step-Back        → 1 個通用查詢  │
└───────────────────────────────────┘
        │
        ▼
向量搜尋（可能是多次）
        │
        ▼
RRF 合併 + 去重
        │
        ▼
LLM 生成最終答案
```

## 運作原理

### 技術一：Query Rewriting（查詢重寫）

最直接的方式：讓 LLM 把使用者的口語查詢改寫成更適合向量搜尋的形式。

**原始查詢**：「怎麼讓模型不要亂說話」  
**改寫後**：「大型語言模型幻覺抑制（Hallucination Mitigation）技術有哪些？」

適合場景：查詢過於口語、過短、或包含歧義詞時。

---

### 技術二：Multi-Query（多查詢）

對原始查詢，從 3-5 個不同角度生成子查詢，各自進行向量搜尋，合併所有結果。

**原始查詢**：「RAG 的記憶體問題」

**生成的子查詢**：
1. 「RAG 系統的記憶體使用量如何優化？」
2.「大規模向量資料庫的記憶體佔用分析」
3.「LangChain RAG pipeline 記憶體洩漏排查」
4.「embedding 模型批次處理記憶體管理」
5.「向量搜尋 FAISS vs Chroma 記憶體比較」

每個子查詢觸達不同的文件叢集，合併後覆蓋率遠超原始單一查詢。

---

### 技術三：HyDE（Hypothetical Document Embeddings）

不直接搜尋「問題」，而是讓 LLM 先**假設性地生成一個答案**，再用這個假設答案去搜尋。

原理：向量空間中，「正確答案的 embedding」和「相關文件的 embedding」更接近，比原始問題的 embedding 更接近目標文件。

```
查詢 → LLM 生成假設答案 → 假設答案 Embedding → 向量搜尋 → 真實文件
```

**HyDE 詳細介紹請見文章 16**，本篇僅作概念說明。

---

### 技術四：Step-Back Prompting（退一步提問）

遇到具體問題時，先「退後一步」問一個更通用的問題，檢索通用問題的答案，再作為 context 回答具體問題。

**具體問題**：「GPT-4 的 context window 是多少 token？」  
**退後一步**：「主流大型語言模型的 context window 規格如何？」

搜尋通用問題的結果通常包含更完整的背景知識，讓 LLM 在回答具體問題時不容易出錯。

---

### RRF（Reciprocal Rank Fusion）：合併多查詢結果

Multi-Query 產生多組搜尋結果，需要一個合理的方式合併並重新排序。RRF 是目前最廣泛使用的方法：

$$\text{RRF}(d) = \sum_{i=1}^{n} \frac{1}{k + r_i(d)}$$

其中 $r_i(d)$ 是文件 $d$ 在第 $i$ 次搜尋結果中的排名，$k$ 通常設為 60。

**直觀理解**：每份文件從每組排名中獲得一個「倒數分數」，在所有查詢結果中都排名靠前的文件得分最高，即使它在任一組中都不是第一名。

## Python 實作範例

以下展示 LangChain `MultiQueryRetriever` 完整實作，並加入自訂 RRF 合併邏輯。

```python
import os
from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser


# ── 1. 自訂多查詢輸出解析器 ──

class LineListOutputParser(BaseOutputParser[List[str]]):
    """將 LLM 輸出的多行文字解析為查詢列表。"""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        # 過濾空行，並去除行首的數字和標點
        queries = []
        for line in lines:
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                queries.append(line)
        return queries


# ── 2. 初始化元件 ──

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

# 建立向量庫（實際使用時替換為你的文件）
sample_texts = [
    "向量資料庫的記憶體管理：FAISS 使用 mmap 技術讓大型索引不需全部載入記憶體。",
    "幻覺抑制技術：Chain-of-Thought、自我一致性、及 RAG 都能有效減少 LLM 幻覺。",
    "LangChain RAG pipeline 最佳實踐：使用 async 批次 embedding，避免記憶體洩漏。",
    "大型語言模型 context window 比較：GPT-4 128K、Claude 3 200K、Gemini 1M tokens。",
    "向量搜尋效能優化：批次查詢、快取 embedding、使用 HNSW 索引代替暴力搜尋。",
]

vectorstore = Chroma.from_texts(
    texts=sample_texts,
    embedding=embeddings,
    collection_name="rag_demo"
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── 3. 設計多查詢生成 Prompt ──

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一個擅長資訊檢索的 AI 助理。
使用者提出了以下問題：
{question}

請從 4 個不同角度重新表達這個問題，以便在向量資料庫中找到更多相關文件。
每個角度應該：
- 使用不同的技術術語或同義詞
- 關注問題的不同面向
- 保持問題的核心意圖

請直接輸出 4 個問題，每行一個，不需要編號或前綴："""
)


# ── 4. 建立 MultiQueryRetriever ──

output_parser = LineListOutputParser()

retriever = MultiQueryRetriever(
    retriever=base_retriever,
    llm_chain=MULTI_QUERY_PROMPT | llm | output_parser,
    include_original=True  # 也包含原始查詢的搜尋結果
)


# ── 5. 實作 RRF 合併函數 ──

def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    對多組搜尋結果執行 Reciprocal Rank Fusion。
    
    Args:
        results_list: 每個子查詢的搜尋結果列表
        k: RRF 平滑常數，通常設 60
    
    Returns:
        按 RRF 分數排序的 (Document, score) 列表
    """
    # 用文件內容作為唯一識別
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_objects: Dict[str, Document] = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content  # 實際使用時應改為文件 ID
            doc_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_objects[doc_id] = doc
    
    # 按分數降序排列
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_objects[doc_id], score) for doc_id, score in sorted_docs]


# ── 6. Step-Back Prompting 實作 ──

STEP_BACK_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""給定以下具體問題，請生成一個更通用、更抽象的「退後一步」問題。
這個通用問題應該能提供更廣泛的背景知識，幫助回答原始問題。

具體問題：{question}

退後一步的通用問題（只輸出問題本身）："""
)

def step_back_retrieval(query: str, top_k: int = 5) -> List[Document]:
    """
    Step-Back Prompting：先生成通用問題，再結合原始問題進行檢索。
    """
    # 生成退後一步的通用問題
    step_back_chain = STEP_BACK_PROMPT | llm
    step_back_question = step_back_chain.invoke({"question": query}).content
    print(f"  Step-Back 問題：{step_back_question}")
    
    # 分別檢索原始問題和通用問題
    original_docs = base_retriever.invoke(query)
    stepback_docs = base_retriever.invoke(step_back_question)
    
    # 合併結果
    fused = reciprocal_rank_fusion([original_docs, stepback_docs])
    return [doc for doc, _ in fused[:top_k]]


# ── 7. 完整 Multi-Query 檢索流程 ──

def multi_query_retrieval(query: str) -> List[Tuple[Document, float]]:
    """
    使用 MultiQueryRetriever 獲取所有子查詢的結果，
    然後用 RRF 合併排序。
    """
    print(f"\n原始查詢：{query}")
    
    # MultiQueryRetriever 內部已處理多查詢，直接返回去重後結果
    # 若需要 RRF，我們手動呼叫每個子查詢
    
    # 生成子查詢
    queries_chain = MULTI_QUERY_PROMPT | llm | output_parser
    sub_queries = queries_chain.invoke({"question": query})
    sub_queries.append(query)  # 加入原始查詢
    
    print(f"生成的子查詢：")
    for i, q in enumerate(sub_queries):
        print(f"  {i+1}. {q}")
    
    # 每個子查詢各自搜尋
    all_results = []
    for q in sub_queries:
        docs = base_retriever.invoke(q)
        all_results.append(docs)
    
    # RRF 合併
    fused_results = reciprocal_rank_fusion(all_results)
    
    print(f"\nRRF 合併後前 3 筆：")
    for doc, score in fused_results[:3]:
        print(f"  [分數: {score:.4f}] {doc.page_content[:60]}...")
    
    return fused_results


# ── 使用範例 ──
if __name__ == "__main__":
    # 測試 Multi-Query + RRF
    results = multi_query_retrieval("RAG 記憶體問題怎麼解決")
    
    print("\n" + "="*50)
    
    # 測試 Step-Back Prompting
    print("\nStep-Back Prompting 示範：")
    docs = step_back_retrieval("GPT-4 的 context window 是多少？")
    for doc in docs[:2]:
        print(f"  → {doc.page_content[:80]}...")
```

**安裝依賴**：
```bash
pip install langchain langchain-openai langchain-chroma chromadb
```

## 優缺點分析

### 優點

**1. 顯著提升召回率**：Multi-Query 在多個評測資料集上將召回率提升 20-40%，對於模糊查詢效果尤其顯著。

**2. 無需修改索引**：所有改造都在查詢端進行，不需要重建向量庫，可以立即套用到現有系統。

**3. 技術組合靈活**：Query Rewriting、Multi-Query、Step-Back 可以獨立使用，也可以組合，根據業務場景調整。

**4. RRF 合併魯棒性高**：Reciprocal Rank Fusion 對各組排名的噪音不敏感，不需要分數校準即可使用。

### 缺點

**1. 增加 LLM 呼叫次數**：Multi-Query 每次查詢需要額外 1 次 LLM 呼叫生成子查詢，加上每個子查詢各自搜尋，整體延遲可能增加 2-5 倍。

**2. 子查詢品質依賴 Prompt**：生成的子查詢若偏離原意，反而會引入無關文件，降低精準率。

**3. 成本線性增加**：3 個子查詢 = 3 倍向量搜尋成本（對自建 FAISS 影響較小，對雲端向量庫費用有影響）。

**4. 不適合精確查詢**：對於精確的專有名詞查詢（如 「CVE-2024-12345」），多查詢改寫可能反而分散焦點。

## 適用場景

**模糊查詢、口語化問答**：使用者問題缺乏技術術語，和文件語言差距大時，Multi-Query 和 Query Rewriting 都有顯著效果。

**跨主題複合問題**：「比較 A 和 B 的效能差異，並說明什麼時候用 C」——這類問題涉及多個主題，Multi-Query 可拆成多個單主題子查詢，分別精確命中。

**專業知識庫 Q&A**：法律、醫療、財務等領域，使用者習慣用通俗語言描述，而知識庫使用專業術語。Step-Back Prompting 可以幫助橋接術語差距。

**低延遲不敏感的場景**：如果 API 呼叫時間增加 1-2 秒可以接受（非即時互動介面），Multi-Query 幾乎沒有部署門檻。

## 與其他方法的比較

| 方法 | 改造對象 | 主要效益 | 延遲影響 | 適合問題類型 |
|------|---------|---------|---------|------------|
| **Query Rewriting** | 查詢（1→1） | 精確性↑ | +低 | 口語/模糊查詢 |
| **Multi-Query** | 查詢（1→N） | 召回率↑ | +中 | 複合/寬泛查詢 |
| **HyDE** | 查詢→假設答案 | 語義距離↓ | +中 | 技術性問答 |
| **Step-Back** | 查詢→通用問題 | 背景知識↑ | +低 | 需要上下文的具體問題 |
| **Document Augmentation** | 文件索引 | 召回率↑ | 無（預處理） | 文件語義單一 |

**Query Transformation vs. Document Augmentation**：本質都是解決語義不匹配，但一個在查詢端動刀，一個在索引端動刀。實務中常常兩者一起用——索引時做 Document Augmentation，查詢時做 Multi-Query，效果疊加。

## 小結

Query Transformation 是 RAG 系統中投入產出比最高的優化方向之一：不需要修改索引，只需在查詢管道加幾行程式碼，就能顯著提升召回率。

四種技術的推薦使用順序：

1. **先上 Query Rewriting**：成本最低，對口語查詢幾乎有即時效果。
2. **加入 Multi-Query + RRF**：當 Rewriting 仍然遺漏文件時，Multi-Query 能覆蓋更多語義面向。
3. **特定場景試 Step-Back**：問題需要廣泛背景知識支撐時（如解釋原理、比較優劣）。
4. **HyDE 留給深度優化階段**：效果強但需要仔細調校，見文章 16 的完整討論。

若延遲是瓶頸，可以將子查詢改為 async 並行執行，把 4 次序列呼叫壓縮成 1 次並行，延遲幾乎回到單查詢水準。
