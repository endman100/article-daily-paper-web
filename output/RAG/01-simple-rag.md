---
title: "Simple RAG：用向量搜尋讓 LLM 說出你的資料"
description: "從零實作基礎 RAG 系統，涵蓋文件切分、向量化、FAISS 儲存到 LangChain 查詢全流程，並分析其優缺點與適用場景，作為 RAG 系列的起點。"
date: 2025-01-15
tags: ["RAG", "LangChain", "FAISS", "OpenAI", "向量資料庫", "LLM", "NLP"]
---

# Simple RAG：用向量搜尋讓 LLM 說出你的資料

你問 GPT-4「我們公司 Q3 的銷售數字是多少？」，它不知道。你問它「昨天發布的 CVE 漏洞細節？」，它也不知道。這不是模型能力的問題，而是架構問題——模型的知識在訓練截止日後就凍結了，它根本看不到你的私有資料。

RAG（Retrieval-Augmented Generation）解決的就是這個問題：在回答問題之前，先從外部知識庫撈出相關文件片段，把它們塞進 prompt，讓 LLM 基於這些真實資料回答。Simple RAG 是最基礎的實作形式，理解它是掌握所有進階變體的前提。

---

## 核心概念

傳統 LLM 的兩個根本限制：

**1. 知識截止日期（Knowledge Cutoff）**
模型訓練資料有固定的截止點。GPT-4o 的截止日是 2023 年 10 月，之後發生的事它一概不知。更嚴重的是，你的內部文件、產品手冊、客戶合約，從來就不在訓練資料裡。

**2. 幻覺問題（Hallucination）**
當模型不知道答案卻被迫回答時，它會「創造」一個聽起來合理的答案。這在企業應用中是災難性的——錯誤的法律條文、捏造的產品規格、虛假的數據引用。

RAG 的解法是在推論階段（inference time）動態注入知識，而非重新訓練模型：

```
問題 → 從知識庫檢索相關文件 → 將文件 + 問題組成 prompt → LLM 生成有依據的回答
```

這樣做的本質是把「記憶」外部化——LLM 負責推理和語言生成，知識庫負責儲存和檢索事實。

---

## 運作原理

Simple RAG 分為兩個階段：**索引階段**（離線）和**查詢階段**（線上）。

```
╔══════════════════════════════════════════════════════════════╗
║                     索引階段（離線）                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  原始文件（PDF/TXT/網頁）                                     ║
║        │                                                     ║
║        ▼                                                     ║
║  ┌─────────────────┐                                         ║
║  │  Document Loader │  載入並解析各種格式的文件               ║
║  └────────┬────────┘                                         ║
║           │                                                  ║
║           ▼                                                  ║
║  ┌─────────────────┐                                         ║
║  │  Text Splitter  │  切分成固定大小的 chunk（例如 500 字）   ║
║  └────────┬────────┘                                         ║
║           │                                                  ║
║           ▼                                                  ║
║  ┌─────────────────┐                                         ║
║  │   Embeddings    │  將每個 chunk 轉換為向量（例如 1536 維） ║
║  └────────┬────────┘                                         ║
║           │                                                  ║
║           ▼                                                  ║
║  ┌─────────────────┐                                         ║
║  │   VectorStore   │  將向量儲存到 FAISS / Chroma / Pinecone ║
║  └─────────────────┘                                         ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                     查詢階段（線上）                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  使用者問題："公司的退費政策是什麼？"                         ║
║        │                                                     ║
║        ▼                                                     ║
║  ┌─────────────────┐                                         ║
║  │   Embeddings    │  將問題也轉換為向量                     ║
║  └────────┬────────┘                                         ║
║           │                                                  ║
║           ▼                                                  ║
║  ┌─────────────────┐                                         ║
║  │    Retriever    │  餘弦相似度搜尋，取出 Top-K 個 chunk    ║
║  └────────┬────────┘                                         ║
║           │                                                  ║
║           ▼                                                  ║
║  ┌─────────────────────────────────────────┐                 ║
║  │  Prompt = 問題 + Retrieved Context      │                 ║
║  │  "根據以下資料回答：[chunk1][chunk2]..." │                 ║
║  └────────┬────────────────────────────────┘                 ║
║           │                                                  ║
║           ▼                                                  ║
║  ┌─────────────────┐                                         ║
║  │       LLM       │  生成最終回答                           ║
║  └─────────────────┘                                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**相似度搜尋的數學基礎**

向量化後，每個文字片段都成為高維空間中的一個點。查詢向量與文件向量的餘弦相似度（Cosine Similarity）越高，代表語意越接近：

```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

FAISS（Facebook AI Similarity Search）使用近似最近鄰（ANN）演算法，讓這個搜尋在百萬級向量中仍能毫秒級完成。

---

## Python 實作範例

```python
# 安裝套件：pip install langchain langchain-openai langchain-community faiss-cpu

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── 設定 API Key ──────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# ══════════════════════════════════════════════════════════════
# 索引階段：建立知識庫
# ══════════════════════════════════════════════════════════════

# ── 步驟 1：載入文件 ──────────────────────────────────────────
# TextLoader 支援 UTF-8 編碼，可換成 PyPDFLoader / WebBaseLoader
loader = TextLoader("company_policy.txt", encoding="utf-8")
documents = loader.load()

print(f"載入文件數量：{len(documents)}")
print(f"文件總字數：{sum(len(doc.page_content) for doc in documents)}")

# ── 步驟 2：切分文件 ──────────────────────────────────────────
# chunk_size：每個片段的最大字元數
# chunk_overlap：相鄰片段的重疊字元數（避免語意在邊界被截斷）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 優先按段落切分
)

chunks = text_splitter.split_documents(documents)

print(f"切分後 chunk 數量：{len(chunks)}")
print(f"第一個 chunk 預覽：\n{chunks[0].page_content[:200]}")

# ── 步驟 3：向量化並儲存到 FAISS ─────────────────────────────
# OpenAIEmbeddings 使用 text-embedding-ada-002 模型
# 每個文字片段會被轉換為 1536 維的浮點數向量
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# FAISS.from_documents 會：
# 1. 對每個 chunk 呼叫 embeddings API
# 2. 將向量建立 FAISS 索引（預設使用 IndexFlatL2）
vectorstore = FAISS.from_documents(chunks, embeddings)

# 將索引儲存到磁碟，下次可直接載入，不需重新向量化
vectorstore.save_local("faiss_index")
print("向量索引已儲存至 ./faiss_index/")

# ══════════════════════════════════════════════════════════════
# 查詢階段：回答問題
# ══════════════════════════════════════════════════════════════

# ── 從磁碟載入已建立的索引 ────────────────────────────────────
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,  # 信任自己建立的索引
)

# ── 步驟 4：建立 Retriever ────────────────────────────────────
# k=3 表示每次查詢取回最相似的 3 個 chunk
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 使用餘弦相似度
    search_kwargs={"k": 3},
)

# ── 步驟 5：初始化 LLM ────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # 設為 0 讓回答更確定、減少創造性幻覺
)

# ── 步驟 6：自訂 Prompt 模板 ──────────────────────────────────
# 明確告訴模型只能依據提供的資料回答，不得自行捏造
prompt_template = """你是一個專業的客服助理，請根據以下提供的資料回答問題。
如果資料中沒有相關資訊，請明確說明「根據現有資料無法回答此問題」，不要自行推測。

參考資料：
{context}

問題：{question}

回答："""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

# ── 步驟 7：組合成 RetrievalQA Chain ──────────────────────────
# chain_type="stuff"：將所有 retrieved chunks 直接填入 prompt
# 其他選項：map_reduce（分別處理後彙整）、refine（逐步精煉）
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,  # 回傳來源文件，方便 debug
    chain_type_kwargs={"prompt": PROMPT},
)

# ── 執行查詢 ──────────────────────────────────────────────────
def ask(question: str) -> None:
    """查詢知識庫並印出回答與來源"""
    result = qa_chain.invoke({"query": question})

    print(f"\n{'='*60}")
    print(f"問題：{question}")
    print(f"{'='*60}")
    print(f"回答：\n{result['result']}")
    print(f"\n{'─'*40}")
    print("參考來源：")
    for i, doc in enumerate(result["source_documents"], 1):
        # 顯示來源文件的前 100 字和 metadata
        source = doc.metadata.get("source", "未知來源")
        print(f"  [{i}] {source}：{doc.page_content[:100]}...")


# 實際查詢範例
ask("公司的退費政策是什麼？")
ask("如何申請年假？")
ask("IT 設備損壞的處理流程？")

# ── 進階：直接查看相似度分數 ──────────────────────────────────
def search_with_score(query: str, k: int = 3) -> None:
    """顯示每個 chunk 的相似度分數（越低越相似，L2 距離）"""
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    print(f"\n查詢：'{query}' 的相似度搜尋結果")
    for doc, score in docs_with_scores:
        print(f"  分數 {score:.4f}：{doc.page_content[:80]}...")


search_with_score("退費申請")
```

**執行前準備**

建立一個測試用的 `company_policy.txt`：

```
公司退費政策
購買後 30 天內可申請全額退費，需附上原始發票。
30 天後至 90 天內可申請 50% 退費。
超過 90 天恕不受理退費申請。
退費申請請寄至 refund@company.com，處理時間為 5-7 個工作日。

年假申請流程
員工每年享有 14 天特休假。
年假申請需提前 3 天透過 HR 系統提交。
緊急情況可事後補填，但需主管簽核。
```

---

## 優缺點分析

| 面向 | 優點 | 缺點 |
|------|------|------|
| **實作複雜度** | 架構簡單，20 行核心程式碼即可運作 | — |
| **可維護性** | 新增/更新知識只需重新索引，無需重新訓練 | 大規模文件更新時索引成本高 |
| **準確性** | 回答有明確依據，可追溯來源 | 相似度搜尋≠語意理解，關鍵字不重疊時召回率低 |
| **Chunk 邊界** | — | 語意可能在切分點斷裂，導致 context 不完整 |
| **查詢能力** | 單一明確問題效果佳 | 需要跨 chunk 推理的複雜問題（如比較、彙整）表現差 |
| **延遲** | 向量搜尋極快（毫秒級） | 每次查詢需額外的 embedding API 呼叫 |
| **成本** | 索引建立一次性費用低 | 高頻查詢的 embedding + LLM API 費用累積 |
| **多語言** | OpenAI embeddings 原生支援中文 | 跨語言查詢（中文問英文文件）效果不穩定 |

**最常遇到的失敗模式：**

- **同義詞問題**：問「費用退還」但文件寫「退費政策」，語意相近但向量距離可能較遠。
- **Chunk 截斷**：一個完整的條款被切成兩半，每半邊單獨看都沒有完整意義。
- **Top-K 不足**：`k=3` 可能剛好把最相關的第 4 個 chunk 排除在外。
- **幻覺殘留**：即使有 context，LLM 仍可能忽略它並從訓練資料回答。

---

## 適用場景

**最適合 Simple RAG 的情境：**

**1. FAQ 問答系統**
結構化的問答對，每個問題有明確唯一的答案。用戶輸入的問題通常與 FAQ 問題高度相似，向量搜尋召回率高。電商平台的「配送說明」、「退換貨流程」是典型案例。

**2. 內部知識庫搜尋**
企業內部文件（HR 政策、產品手冊、IT 規範），文件穩定、不頻繁更新。員工提問通常是事實查詢，不需要跨文件推理。

**3. 文件問答系統**
對特定文件（合約、報告、論文）提問。文件範圍明確，相關 chunk 集中，Simple RAG 召回精準。法律事務所的合約審查助理是典型應用。

**4. 客服輔助**
輔助客服人員快速找到標準回答腳本，不是直接面向終端用戶的自動化回答。人工確認環節可以彌補召回不精準的問題。

**不適合的情境：**
- 需要彙整多份文件才能回答的問題（如「所有關於退費的規定」）
- 問題答案分散在 10 個以上不同 chunk
- 需要數值計算或邏輯推理的問題
- 文件更新頻率極高（每日數千份）

---

## 與其他方法的比較

| 方法 | 原理 | 適用規模 | 延遲 | 複雜度 | 適合場景 |
|------|------|----------|------|--------|----------|
| **Simple RAG** | 單次向量搜尋 + LLM | 中小型知識庫 | 低 | ★☆☆☆☆ | 明確事實查詢 |
| **Hybrid RAG** | 向量搜尋 + BM25 關鍵字搜尋融合 | 中大型 | 中 | ★★☆☆☆ | 專有名詞多的領域 |
| **HyDE** | 先讓 LLM 生成假設答案，再用答案搜尋 | 中型 | 中高 | ★★☆☆☆ | 抽象概念查詢 |
| **Multi-Query RAG** | 將問題改寫成多個查詢，合併結果 | 中型 | 高 | ★★★☆☆ | 複雜問題分解 |
| **Self-RAG** | LLM 自主決定是否需要檢索 | 任意 | 高 | ★★★★☆ | 需要推理的問題 |
| **Graph RAG** | 知識圖譜 + 向量搜尋 | 大型複雜 | 高 | ★★★★★ | 多跳推理、關係查詢 |
| **Fine-tuning** | 將知識烘焙進模型權重 | — | 極低（無檢索） | ★★★★★ | 靜態、高頻知識 |

**為什麼不直接 Fine-tuning？**

| | Fine-tuning | RAG |
|---|---|---|
| 知識更新 | 重新訓練（數小時到數天） | 重新索引（分鐘級） |
| 可解釋性 | 無法追溯答案來源 | 可以顯示 source document |
| 知識邊界 | 模型仍可能幻覺 | 可強制限制在 context 內 |
| 成本 | 一次性高成本 | 每次查詢有 API 費用 |

兩者不互斥——Fine-tuning 負責讓模型學會格式和風格，RAG 負責注入最新事實知識，組合使用效果最佳。

---

## 小結

Simple RAG 是三個動作的組合：**切**（Text Splitter）、**藏**（VectorStore）、**找**（Retriever），找到後交給 LLM 回答。這個架構在 1 天內就能上線，解決 80% 的企業知識問答需求。

**三個必須調整的參數：**
1. `chunk_size`：太小則 context 不完整，太大則 token 成本高且噪音多。從 300-500 開始試，依文件結構調整。
2. `chunk_overlap`：設為 `chunk_size` 的 10%-15%，防止邊界截斷語意。
3. `k`（Retriever 取回數量）：從 3 開始，若回答品質差先試著增加到 5-10。

**Simple RAG 的限制指向了後續的改進方向：**
- 召回率不精準 → **Hybrid RAG**（加入 BM25 關鍵字搜尋）
- Chunk 邊界問題 → **Sentence Window Retrieval**（擴大回傳的上下文視窗）
- 複雜多跳問題 → **Multi-Query RAG** 或 **Graph RAG**
- 無法彙整全文 → **Map-Reduce Chain**

本系列後續章節將逐一拆解這些進階技術，每一篇都建立在 Simple RAG 的基礎之上。先把這篇的程式碼跑通，是進入後面內容的前提。
