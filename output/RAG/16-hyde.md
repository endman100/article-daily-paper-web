---
title: "HyDE：用假設文件填補查詢與文件的語義鴻溝"
description: "HyDE（Hypothetical Document Embeddings）提出一個反直覺的解法：與其直接用查詢去搜尋文件，不如先讓 LLM 根據查詢「幻想」一個理想答案，再用這個假設答案去搜尋真實文件。這個技術在 zero-shot 場景下的效果可媲美 fine-tuned 模型。"
date: 2024-01-16
tags: ["RAG", "HyDE", "Hypothetical Document Embeddings", "Query Expansion", "Dense Retrieval", "Zero-Shot"]
---

# HyDE：用假設文件填補查詢與文件的語義鴻溝

HyDE（Hypothetical Document Embeddings，假設文件嵌入）來自 Gao et al. 在 2022 年發表的論文《Precise Zero-Shot Dense Retrieval without Relevance Labels》。它的核心洞察非常反直覺卻極為有效：在向量空間中，一個「假設的（可能是錯誤的）答案文件」往往比原始的查詢更接近真實的答案文件。這個技術不需要任何標記資料，在 zero-shot 情況下就能大幅提升檢索品質。

---

## 核心概念

### 查詢與文件的天然鴻溝

在 Dense Retrieval（密集向量檢索）中，有一個長期存在的根本問題：**查詢（Query）和文件（Document）在語義空間中天然存在差距**。

**查詢的特徵**：
- 通常很短（5-20 個詞）
- 是問句形式（「台積電的 3nm 製程有哪些優點？」）
- 描述的是「資訊需求」，而不是「知識本身」

**文件的特徵**：
- 通常很長（幾百到幾千個詞）
- 是陳述句形式（「台積電的 3nm 製程在2022年量產，相比5nm提升了60%的邏輯密度...」）
- 描述的是「知識本身」

即使使用了雙塔（Bi-encoder）模型或經過 fine-tune 的嵌入模型，這個「問答非對稱性」依然存在。特別是在 zero-shot 場景（沒有針對特定領域的標記訓練資料），差距更為明顯。

### HyDE 的核心洞察

HyDE 的假設是：**在語義空間中，「一篇可能回答這個問題的假設文件」比「原始問題本身」更接近真實的相關文件**。

用幾何直覺理解：

```
語義空間（向量空間）

         真實文件 D*
        ●  
       / \
      /   \
假設文件 H  \
    ●        \
     \         \
      \         \
       \         ●
        \      原始查詢 Q
         \
  向量距離：d(H, D*) < d(Q, D*)
```

為什麼假設文件更近？因為假設文件和真實文件都是「陳述句式的文本」，它們有相似的詞彙分佈、句式結構和語義密度，而查詢是「問句式的文本」，形式上就已經不同了。

---

## 運作原理

### 完整流程

```
原始查詢 Q
    │
    ▼
┌──────────────────────────────────────┐
│  步驟 1：LLM 生成假設答案文件 H        │
│                                       │
│  Prompt：「請根據以下問題寫一篇       │
│  簡短的解答文件（100字內）：{Q}」     │
│                                       │
│  輸出：一段假設性的答案文本            │
│  （可能有事實錯誤，但沒關係！）        │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  步驟 2：Embed 假設答案文件            │
│                                       │
│  e(H) = Embedding(假設文件 H)         │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  步驟 3：用 e(H) 搜尋向量資料庫        │
│                                       │
│  找到最相似的真實文件 D1, D2, ...Dk   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  步驟 4：LLM 用真實文件生成最終答案    │
│                                       │
│  輸入：原始查詢 Q + 真實文件 D1..Dk   │
│  輸出：有事實依據的最終回答            │
└──────────────────────────────────────┘
```

### 關鍵細節：多個假設文件取平均

原論文中的一個重要技巧：**生成多個假設文件，取向量平均值**。

```python
# 生成 k 個假設文件
hypothetical_docs = [generate_hypothesis(query) for _ in range(k)]

# 分別嵌入
embeddings = [embed(doc) for doc in hypothetical_docs]

# 取平均（平均向量比任何單一向量都更穩定）
avg_embedding = sum(embeddings) / k

# 用平均向量搜尋
results = vectorstore.similarity_search_by_vector(avg_embedding)
```

取平均的好處：單一的假設文件可能偏差較大，多個假設文件的平均向量更能代表「問題答案應該在的語義空間」。

---

## Python 實作範例

### 完整 HyDE 實作與對比測試

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import HypotheticalDocumentEmbedder
import numpy as np
from typing import List

# ─────────────────────────────────────
# 步驟 1：建立測試知識庫
# ─────────────────────────────────────
sample_documents = [
    Document(
        page_content="""台積電的 3nm 製程（N3）於2022年下半年正式量產，是業界首個量產的3nm級製程。
        相比 5nm（N5），N3 製程的邏輯密度提升約60%，同功耗下效能提升約15%，
        同效能下功耗降低約30%。首批採用 N3 製程的產品是 Apple A17 Pro 和 M3 系列晶片。
        N3 製程仍採用 FinFET 架構，但採用了更密集的鰭片設計和改良的接觸電阻技術。""",
        metadata={"source": "tsmc_process_tech", "topic": "製程技術"}
    ),
    Document(
        page_content="""HBM（High Bandwidth Memory，高頻寬記憶體）是一種使用 3D 堆疊技術的 DRAM 記憶體。
        HBM3 的理論頻寬高達 819 GB/s（每個堆疊），是 GDDR6X 的約4倍。
        NVIDIA H100 GPU 搭載了6個 HBM3 堆疊，總記憶體頻寬達到 3.35 TB/s，
        這對大型 AI 模型的訓練至關重要，因為記憶體頻寬往往是訓練效率的瓶頸。""",
        metadata={"source": "hbm_tech", "topic": "記憶體技術"}
    ),
    Document(
        page_content="""Transformer 架構由 Google 在2017年的論文《Attention Is All You Need》提出。
        其核心是自注意力機制（Self-Attention），允許模型在處理序列時考慮所有位置的相互關係。
        相比 RNN/LSTM，Transformer 可以并行計算，大幅提升訓練效率。
        GPT、BERT、T5 等大型語言模型都以 Transformer 為基礎架構。""",
        metadata={"source": "transformer", "topic": "AI架構"}
    ),
    Document(
        page_content="""向量資料庫（Vector Database）是專門用於儲存和搜尋高維向量的資料庫系統。
        主流向量資料庫包括 Pinecone、Weaviate、Chroma、Milvus 等。
        它們使用近似最近鄰搜尋（ANN）算法（如 HNSW、IVF）來高效處理向量相似度查詢。
        在 RAG 應用中，向量資料庫負責儲存文本的嵌入向量，並在查詢時快速找到最相似的文本片段。""",
        metadata={"source": "vector_db", "topic": "向量資料庫"}
    ),
    Document(
        page_content="""ASML 的 EUV（極紫外光）光刻機是現代先進半導體製程的核心設備。
        EUV 使用 13.5nm 波長的光源（比 DUV 的 193nm 短得多），能實現更精細的圖案刻蝕。
        每台 EUV 機器售價約1.5-2億美元，全球每年產量約50-60台，幾乎全數預購一空。
        沒有 EUV 光刻機，就無法製造 7nm 及以下的先進製程晶片。""",
        metadata={"source": "asml_euv", "topic": "半導體設備"}
    ),
]

# ─────────────────────────────────────
# 步驟 2：建立向量資料庫
# ─────────────────────────────────────
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(sample_documents, embeddings_model)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  # 稍高 temperature 增加假設文件的多樣性

# ─────────────────────────────────────
# 步驟 3：實作 HyDE 核心邏輯
# ─────────────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一個專業的技術文章撰寫者。
根據以下問題，撰寫一段約100-150字的技術性段落，
就像這段文字出現在一篇關於該主題的技術文章中一樣。
不需要回答問題，而是寫一段包含該問題答案的文章段落。
注意：你的回答可能有事實上的不準確，但沒關係，
這只是用於幫助搜尋，最終答案會基於真實文件。"""),
    ("human", "問題：{query}")
])


def generate_hypothetical_doc(query: str, n: int = 3) -> List[str]:
    """
    生成 n 個假設文件。
    多個假設文件可以覆蓋更廣的語義空間，提升搜尋穩定性。
    """
    hypothetical_docs = []
    for i in range(n):
        response = llm.invoke(HYDE_PROMPT.format_messages(query=query))
        hypothetical_docs.append(response.content)
        print(f"\n假設文件 {i+1}：\n{response.content[:200]}...")
    return hypothetical_docs


def hyde_retrieve(query: str, k: int = 3, n_hypothetical: int = 3) -> List[Document]:
    """
    HyDE 檢索流程：
    1. 生成多個假設文件
    2. 對假設文件嵌入並取平均
    3. 用平均向量搜尋真實文件
    """
    print(f"\n{'='*60}")
    print(f"HyDE 檢索：{query}")
    print('='*60)
    
    # 生成假設文件
    hyp_docs = generate_hypothetical_doc(query, n=n_hypothetical)
    
    # 對所有假設文件進行嵌入
    hyp_embeddings = [embeddings_model.embed_query(doc) for doc in hyp_docs]
    
    # 計算平均嵌入向量（核心技巧）
    avg_embedding = np.mean(hyp_embeddings, axis=0).tolist()
    
    # 用平均嵌入向量搜尋真實文件
    results = vectorstore.similarity_search_by_vector(avg_embedding, k=k)
    
    print(f"\n檢索到的文件：")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
    
    return results


def standard_retrieve(query: str, k: int = 3) -> List[Document]:
    """
    標準向量搜尋（不使用 HyDE）。
    直接用查詢的嵌入向量搜尋。
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


# ─────────────────────────────────────
# 步驟 4：使用 LangChain 內建的 HyDE 支援
# ─────────────────────────────────────

def build_langchain_hyde_retriever():
    """
    使用 LangChain 的 HypotheticalDocumentEmbedder
    這是更簡潔的生產級實作方式。
    """
    # HypotheticalDocumentEmbedder 將 LLM 和 Embedder 結合
    hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
        llm=ChatOpenAI(model="gpt-4o", temperature=0.7),
        embeddings=embeddings_model,
        prompt_key="web_search"  # 內建 prompt 模板
    )
    
    # 用 HyDE embedder 建立向量庫（搜尋時自動使用 HyDE）
    hyde_vectorstore = Chroma.from_documents(sample_documents, hyde_embedder)
    return hyde_vectorstore.as_retriever(search_kwargs={"k": 3})


# ─────────────────────────────────────
# 步驟 5：完整 HyDE RAG 管道
# ─────────────────────────────────────

def hyde_rag(query: str) -> str:
    """
    完整的 HyDE RAG 管道：
    查詢 → 假設文件 → 嵌入搜尋 → 真實文件 → LLM 回答
    """
    # 1. HyDE 檢索真實文件
    real_docs = hyde_retrieve(query, k=3)
    
    # 2. 用真實文件生成最終答案
    context = "\n\n".join([doc.page_content for doc in real_docs])
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個技術助手。根據提供的上下文回答問題，如果上下文不包含足夠資訊，請說明。"),
        ("human", "上下文：\n{context}\n\n問題：{query}")
    ])
    
    response = llm.invoke(final_prompt.format_messages(context=context, query=query))
    return response.content


# ─────────────────────────────────────
# 步驟 6：對比實驗
# ─────────────────────────────────────

def compare_with_without_hyde(query: str):
    """
    對比：有 HyDE vs 無 HyDE 的搜尋效果
    """
    print(f"\n{'#'*60}")
    print(f"查詢：{query}")
    print('#'*60)
    
    # 方法 A：標準搜尋（直接用 query 嵌入）
    standard_docs = standard_retrieve(query)
    print("\n【標準搜尋結果】：")
    for i, doc in enumerate(standard_docs):
        print(f"  [{i+1}] (topic: {doc.metadata.get('topic', 'N/A')}) {doc.page_content[:100]}...")
    
    # 方法 B：HyDE 搜尋
    hyde_docs = hyde_retrieve(query, n_hypothetical=2)  # 節省 API 費用，用2個假設文件
    print("\n【HyDE 搜尋結果】：")
    for i, doc in enumerate(hyde_docs):
        print(f"  [{i+1}] (topic: {doc.metadata.get('topic', 'N/A')}) {doc.page_content[:100]}...")


# 執行對比測試
# 測試 1：技術細節問題
compare_with_without_hyde("晶片製造需要哪些關鍵設備和技術？")

# 測試 2：跨領域問題
compare_with_without_hyde("為什麼 AI 訓練需要高頻寬記憶體？")

# 測試 3：完整 RAG 問答
answer = hyde_rag("台積電最新的製程技術有什麼突破？")
print(f"\n最終答案：{answer}")
```

---

## 優缺點分析

### 優點

**1. Zero-Shot 效果媲美 Fine-tuned 模型**
論文中的實驗表明，HyDE 在沒有任何標記資料的情況下，在多個資訊檢索基準上接近甚至超過了需要大量標記訓練的 Fine-tuned 方法。

**2. 通用性強**
不需要針對特定領域準備訓練資料，只需要一個能生成文本的 LLM 和一個 Embedding 模型就能實現。

**3. 改善詞彙差距問題**
原始查詢可能使用不同的詞彙來表達和文件相同的概念（例如：查詢說「效率提升」，文件說「性能優化」），假設文件更可能使用與真實文件相似的學術/技術詞彙。

**4. 實現簡單**
整個 HyDE 可以用少量代碼實現，LangChain 也提供了 `HypotheticalDocumentEmbedder` 的內建支援。

### 缺點

**1. 假設文件偏差導致搜尋惡化**
這是 HyDE 最大的風險。如果 LLM 對問題的理解有偏差，或者生成了方向錯誤的假設文件，反而會比直接搜尋效果更差。特別是在 LLM 不熟悉的垂直領域。

**2. 增加延遲和成本**
需要額外調用 LLM 生成假設文件，這增加了整體管道的延遲（通常 +1-2 秒）和 API 費用。

**3. 假設文件的幻覺問題**
假設文件本身可能包含錯誤的事實。雖然這不影響最終答案（最終答案基於真實文件），但理論上如果假設文件的錯誤剛好把搜尋帶向錯誤的語義空間，就會有問題。

**4. 對 Embedding 模型的依賴**
HyDE 的效果高度依賴 Embedding 模型的品質。如果 Embedding 模型本身品質不佳，HyDE 改善的效果也會打折。

---

## 適用場景

### 最適合 HyDE 的場景

**冷啟動系統（無標記資料）**
新建的 RAG 系統還沒有足夠的用戶查詢歷史來做 Embedding Fine-tuning，HyDE 可以在 zero-shot 情況下提供接近 fine-tuned 的效果。

**開放域問答**
問題範圍廣泛、不限定領域的知識庫問答，HyDE 能有效彌補查詢與文件的語義差距。

**技術文件搜尋**
技術文件中充滿專業術語，一般用戶的查詢語言和文件語言差距大，HyDE 讓 LLM 充當「翻譯」，將用戶語言轉化為更接近技術文件的語言。

**多語言場景**
用一種語言查詢、搜尋另一種語言的文件時，HyDE 可以先生成目標語言的假設文件，繞過跨語言嵌入的困難。

### 不適合的場景

- **延遲敏感應用**：需要額外 LLM 呼叫
- **LLM 完全不懂的超垂直領域**：假設文件品質差，反而有害
- **查詢本身就是文件形式時**：如搜尋相似文件，此時查詢和文件形式相同，不存在語義差距問題

---

## 與其他方法的比較

| 特性 | 標準向量搜尋 | HyDE | Query Expansion | Reranker |
|------|------------|------|----------------|---------|
| 零標記資料 | ✓ | ✓ | ✓ | 需要訓練 |
| 額外 LLM 呼叫 | 無 | 1次 | 1次 | 無 |
| 解決詞彙差距 | 有限 | ✓✓✓ | ✓✓ | ✓ |
| 方向偏差風險 | 低 | 中 | 中 | 無 |
| 計算成本 | 低 | 中 | 中 | 中 |

**HyDE vs Query Expansion（查詢擴展）**：
兩者都使用 LLM 在搜尋前做處理，但目標不同。Query Expansion 是生成多個不同表述的查詢詞，然後合併多個搜尋結果；HyDE 是生成一個假設的答案文件，用它的嵌入向量進行搜尋。實際應用中兩者可以組合使用。

---

## 小結

HyDE 是一個思想上優雅、實作上簡單的技術突破。它提出的核心問題——「為什麼要用問題去找答案，而不是用答案去找答案？」——讓人重新審視 RAG 管道的設計假設。

從工程角度，HyDE 是提升 RAG 搜尋品質成本效益最高的方法之一：只需增加一次 LLM 呼叫，就能在 zero-shot 場景下獲得接近 fine-tuned 的搜尋效果。

在實際部署中，建議將 HyDE 作為 RAG 管道的標準組件，配合適當的 fallback 策略（當假設文件品質不佳時退回到標準搜尋），可以在大多數場景下獲得穩定的效果提升。
