---
title: "Context Enriched RAG：用上下文視窗解決孤立 Chunk 的致命缺陷"
description: "深入解析 Sentence Window Retrieval 與 Parent Document Retrieval 兩種上下文增強策略，附完整 LlamaIndex 與 LangChain Python 實作範例。"
date: 2025-01-17
tags:
  - RAG
  - LlamaIndex
  - LangChain
  - Sentence Window Retrieval
  - Parent Document Retrieval
  - Vector Search
  - LLM
---

# Context Enriched RAG：用上下文視窗解決孤立 Chunk 的致命缺陷

假設你的技術文件裡有這麼一句話：

> 「本方法的準確率比上一章節描述的基準方法高出 23%。」

這句話被切成一個獨立的 chunk，並在向量搜尋中被命中。LLM 拿到這個 chunk 後，根本無從得知「上一章節描述的基準方法」是什麼——它可能是 BERT、TF-IDF、還是某個自訂演算法。LLM 只能猜，或者給出「根據提供的資訊無法確定」這種廢話回答。

這就是標準 RAG 的核心問題：**向量搜尋找的是語意相似度，但切出來的 chunk 往往把關鍵上下文留在了邊界外**。Context Enriched RAG 正是為了修補這個缺陷而設計的。

---

## 核心概念

標準 RAG 的 chunk 策略假設每個文字片段都是自給自足的資訊單元。這個假設在現實中幾乎從不成立：

- 代詞（它、該方法、上述結果）指向前文定義的實體
- 對比句（「然而」、「相較之下」）依賴前句的基準
- 步驟性說明（「接著」、「完成後」）依賴前一個步驟的狀態
- 數據引用（「高出 23%」）依賴前文建立的比較基準

Context Enriched RAG 的解法很直接：**不改變向量索引的精準度，但在把內容傳給 LLM 時，擴展成包含周圍文脈的版本**。

這個方向衍生出兩種主流策略：
1. **Sentence Window Retrieval**（句子視窗檢索）— 以句子為索引單位，回傳時擴展成鄰近句子群
2. **Parent Document Retrieval**（父文件檢索）— 以小 chunk 做向量搜尋，回傳時替換成對應的大 chunk 或原始文件

---

## 運作原理

### Sentence Window Retrieval

```
索引階段：
┌──────────────────────────────────────────────────────────┐
│  原始段落                                                  │
│  S1: 傳統方法使用 TF-IDF 做為基準。                       │
│  S2: 本研究提出改良的注意力機制。                         │
│  S3: 本方法的準確率比基準方法高出23%。  ← 最相似句子     │
│  S4: 此結果在三個資料集上均可重現。                       │
│  S5: 未來工作將探索多語言場景。                           │
└──────────────────────────────────────────────────────────┘

每一句單獨建立向量索引，同時在 metadata 中記錄「視窗範圍」

檢索階段（k=1，視窗大小=2）：
查詢 → 向量搜尋 → 命中 S3

回傳給 LLM 的內容（視窗擴展）：
┌──────────────────────────────────────────────────────────┐
│  S1: 傳統方法使用 TF-IDF 做為基準。         ← 前2句     │
│  S2: 本研究提出改良的注意力機制。           ← 前1句     │
│  S3: 本方法的準確率比基準方法高出23%。      ← 命中句子  │
│  S4: 此結果在三個資料集上均可重現。         ← 後1句     │
│  S5: 未來工作將探索多語言場景。             ← 後2句     │
└──────────────────────────────────────────────────────────┘
```

LLM 現在知道「基準方法」是 TF-IDF，可以給出有意義的回答。

---

### Parent Document Retrieval

```
索引階段：
┌─────────────────────────────────────────────────────────────┐
│  父文件（大 chunk 或整個段落）                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 子 chunk A（小）→ 建立向量索引                       │    │
│  │ 子 chunk B（小）→ 建立向量索引  ← 向量搜尋命中       │    │
│  │ 子 chunk C（小）→ 建立向量索引                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

檢索階段：
查詢 → 小 chunk 向量搜尋 → 命中子 chunk B
     → 查找子 chunk B 的 parent_id
     → 取出完整父文件

回傳給 LLM 的內容：
┌─────────────────────────────────────────────────────────────┐
│  完整父文件（包含 chunk A + B + C 及其周圍所有文字）        │
└─────────────────────────────────────────────────────────────┘
```

**兩種策略的關鍵差異：**

| 面向 | Sentence Window | Parent Document |
|------|----------------|-----------------|
| 向量索引單位 | 單句 | 小 chunk（如 128 tokens） |
| 回傳內容 | 命中句 ± k 個鄰近句 | 對應的父文件（大 chunk） |
| 上下文邊界 | 動態（由 k 決定） | 固定（由父文件切割決定） |
| 適合文件類型 | 句子密度高的長文 | 有明確章節結構的文件 |
| 實作複雜度 | 較低 | 較高（需維護兩層索引） |

---

## Python 實作範例

### 方法一：LlamaIndex Sentence Window Retrieval

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ── 設定 LLM 與 Embedding 模型 ────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "your-api-key"

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ── 載入文件 ──────────────────────────────────────────────────────────────────
# 將你的 PDF / TXT 文件放在 ./docs 目錄下
documents = SimpleDirectoryReader("./docs").load_data()

# ── 建立 SentenceWindowNodeParser ────────────────────────────────────────────
# window_size=3 表示：命中句子的前後各 3 句都會被放入 metadata["window"]
# 向量索引只用單句建立，但 metadata 中保留完整視窗內容
sentence_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                        # 前後各取 3 句
    window_metadata_key="window",         # 視窗內容存在此 metadata key
    original_text_metadata_key="original_sentence",  # 原始命中句子
)

# ── 解析文件為 nodes ──────────────────────────────────────────────────────────
nodes = sentence_node_parser.get_nodes_from_documents(documents)

# 查看 node 結構（除錯用）
print(f"總 node 數量: {len(nodes)}")
print("--- 第一個 node ---")
print(f"索引文字（單句）: {nodes[0].text[:80]}...")
print(f"視窗內容（多句）: {nodes[0].metadata.get('window', '')[:200]}...")

# ── 建立向量索引 ──────────────────────────────────────────────────────────────
# 注意：向量是用 node.text（單句）建立的，確保搜尋精準度
index = VectorStoreIndex(nodes)

# ── 建立 Query Engine，加入 MetadataReplacementPostProcessor ─────────────────
# MetadataReplacementPostProcessor 的作用：
#   向量搜尋命中某個 node 後，將 node.text（單句）
#   替換成 metadata["window"]（視窗多句），再傳給 LLM
query_engine = index.as_query_engine(
    similarity_top_k=3,                   # 先取前 3 個最相似的 node
    node_postprocessors=[
        # 關鍵步驟：用視窗內容取代單句內容
        MetadataReplacementPostProcessor(
            target_metadata_key="window"  # 指定要替換的 metadata key
        ),
        # 可選：用 cross-encoder reranker 進一步排序
        # SentenceTransformerRerank(
        #     model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        #     top_n=2
        # ),
    ],
)

# ── 執行查詢 ──────────────────────────────────────────────────────────────────
query = "本研究提出的方法相較於基準方法有多少改進？"
response = query_engine.query(query)

print("\n=== 查詢結果 ===")
print(f"問題: {query}")
print(f"回答: {response}")

print("\n=== 實際傳給 LLM 的 source nodes（包含視窗擴展後的內容）===")
for i, node in enumerate(response.source_nodes):
    print(f"\n[Node {i+1}] 相似度分數: {node.score:.4f}")
    print(f"視窗內容:\n{node.node.text[:300]}...")
```

---

### 方法二：LangChain Parent Document Retrieval

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

# ── 載入文件 ──────────────────────────────────────────────────────────────────
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# ── 定義兩層切割策略 ──────────────────────────────────────────────────────────
# 子 splitter（小 chunk）：用於建立向量索引，確保搜尋精準
# chunk 越小，向量搜尋越精準，但需要更多的 parent lookup
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # 小 chunk：~400 字元，向量搜尋精準
    chunk_overlap=20,
)

# 父 splitter（大 chunk）：回傳給 LLM，提供完整上下文
# chunk 越大，LLM 拿到的資訊越完整，但 token 消耗越多
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,         # 大 chunk：~2000 字元，保留上下文
    chunk_overlap=100,
)

# ── 建立向量資料庫（只存小 chunk 的向量）────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=embeddings,
)

# ── 建立父文件儲存（存放大 chunk 的原始文字）────────────────────────────────
# InMemoryStore 適合原型開發；生產環境建議用 RedisStore 或 MongoDBStore
docstore = InMemoryStore()

# ── 建立 ParentDocumentRetriever ─────────────────────────────────────────────
# 內部邏輯：
#   1. 用 parent_splitter 切出父文件，存入 docstore（有唯一 ID）
#   2. 用 child_splitter 切出子 chunk，在 metadata 中記錄 parent_id
#   3. 將子 chunk 的向量存入 vectorstore
#   4. 查詢時：小 chunk 向量搜尋 → 取得 parent_id → 從 docstore 取父文件
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    # search_kwargs={"k": 4},  # 向量搜尋取前 k 個小 chunk
)

# ── 索引文件 ──────────────────────────────────────────────────────────────────
retriever.add_documents(documents)

# 驗證索引數量
print(f"向量庫中的小 chunk 數量: {vectorstore._collection.count()}")

# ── 直接使用 retriever 測試 ───────────────────────────────────────────────────
query = "本研究提出的方法相較於基準方法有多少改進？"
retrieved_docs = retriever.get_relevant_documents(query)

print(f"\n查詢: {query}")
print(f"取回的父文件數量: {len(retrieved_docs)}")
for i, doc in enumerate(retrieved_docs):
    print(f"\n[父文件 {i+1}] 長度: {len(doc.page_content)} 字元")
    print(f"內容預覽: {doc.page_content[:200]}...")

# ── 整合進 QA Chain ───────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",       # 將所有父文件一次性 stuff 進 prompt
    retriever=retriever,
    return_source_documents=True,
)

result = qa_chain.invoke({"query": query})
print("\n=== QA Chain 結果 ===")
print(f"回答: {result['result']}")
```

---

### 兩種方法的行為對比（可執行驗證）

```python
# 用一個具體例子驗證「有無上下文增強」的差異
sample_text = """
第三章：實驗設定
本研究採用 BERT-base 作為基準模型（Baseline），在標準 SQuAD 2.0 資料集上
進行評估，F1 分數為 76.3%。

第四章：改良方法
本研究在 BERT-base 的基礎上，引入動態注意力遮罩機制（Dynamic Attention Mask）。
本方法的準確率比上一章節描述的基準方法高出 23%，最終 F1 分數達到 93.9%。
此結果在五次獨立實驗中均保持穩定，標準差為 ±0.4%。
"""

# 模擬標準 RAG（只取命中句子）
standard_chunk = "本方法的準確率比上一章節描述的基準方法高出 23%，最終 F1 分數達到 93.9%。"

# 模擬 Context Enriched RAG（句子視窗，window_size=2）
enriched_window = """
本研究在 BERT-base 的基礎上，引入動態注意力遮罩機制（Dynamic Attention Mask）。
本方法的準確率比上一章節描述的基準方法高出 23%，最終 F1 分數達到 93.9%。
此結果在五次獨立實驗中均保持穩定，標準差為 ±0.4%。
"""

print("標準 RAG 給 LLM 的內容：")
print(standard_chunk)
print("\nLLM 無法知道：「上一章節的基準方法」= BERT-base，F1 = 76.3%")

print("\n" + "="*60)
print("\nContext Enriched RAG 給 LLM 的內容：")
print(enriched_window)
print("\n加入父文件後 LLM 知道：基準 = BERT-base (76.3%)，改進後 = 93.9%")
```

---

## 優缺點分析

| 面向 | Sentence Window Retrieval | Parent Document Retrieval |
|------|--------------------------|--------------------------|
| **搜尋精準度** | ✅ 極高（單句向量） | ✅ 高（小 chunk 向量） |
| **上下文完整性** | ⚠️ 取決於 window_size | ✅ 完整父文件 |
| **Token 消耗** | ⚠️ 可控（window_size 決定） | ⚠️ 較高（父文件通常較大） |
| **索引複雜度** | ✅ 單層索引，metadata 擴展 | ❌ 雙層索引，需維護 docstore |
| **跨段落引用** | ❌ 無法跨越父文件邊界 | ⚠️ 需要父文件切割恰當 |
| **調整彈性** | ✅ 只需調 window_size | ⚠️ 需同時調整兩個 splitter |
| **生產環境成熟度** | ✅ LlamaIndex 原生支援 | ✅ LangChain 原生支援 |
| **冷啟動成本** | 低 | 中（需建立兩個儲存層） |

**主要缺點共通點：**
- 兩種方法都會**增加傳給 LLM 的 token 數量**，直接影響成本與延遲
- 如果父文件切割邊界不當，關鍵上下文仍可能被切斷
- 不解決「相關文件根本不在索引中」的問題（這屬於不同層面的問題）

---

## 適用場景

**Sentence Window Retrieval 最適合：**
- 學術論文、技術報告（充滿代詞和跨句引用）
- 法律文件（條款之間高度相互依賴）
- 醫療記錄（症狀、診斷、處方之間有敘述順序）
- 任何「一句話說不完整」的密集文字文件

**Parent Document Retrieval 最適合：**
- 有明確章節結構的產品文件、API 文件
- FAQ 庫（每個 FAQ 項目自成一個父文件）
- 新聞文章、部落格文章（段落之間有邏輯結構）
- 需要同時回傳「精確片段」和「完整段落」的場景

**兩種方法都不適合：**
- 純表格、結構化數據（上下文增強效果有限）
- 每個 chunk 本身已是完整資訊單元的場景（如短句 Q&A 對）
- Token 預算極度嚴格的場景（上下文增強必然增加 token 數）

---

## 與其他方法的比較

| 方法 | 解決的問題 | 索引策略 | 回傳策略 | 額外成本 |
|------|-----------|---------|---------|---------|
| **標準 RAG** | 基本文件搜尋 | 固定大小 chunk | 直接回傳 chunk | 無 |
| **Sentence Window** | 孤立句子缺乏上下文 | 單句向量 | chunk + 前後 k 句 | 小（metadata 儲存） |
| **Parent Document** | chunk 邊界切斷上下文 | 小 chunk 向量 | 對應父文件 | 中（雙層索引） |
| **HyDE** | 查詢與文件語意差距 | 標準 chunk | 直接回傳 chunk | 中（需 LLM 生成假設文件） |
| **Reranking** | 向量搜尋精準度不足 | 標準 chunk | 重排後 top-k | 中（需 reranker 模型） |
| **Graph RAG** | 實體關係無法表達 | 知識圖譜 | 子圖 + chunk | 高（建圖成本） |

Context Enriched RAG 與 Reranking 是互補的：先用 Sentence Window 或 Parent Document 擴展上下文，再用 cross-encoder reranker 排序，通常能達到比單獨使用任一方法更好的效果（LlamaIndex 的範例程式碼中已有注釋示範此組合）。

---

## 小結

Context Enriched RAG 解決的是一個明確且可量化的問題：**向量搜尋命中的 chunk 因為邊界切割，遺失了讓 LLM 理解所需的上下文**。

兩種策略的選擇原則很直接：
- **文件以句子為邏輯單位**（學術論文、法律文件）→ 選 Sentence Window Retrieval
- **文件以段落或章節為邏輯單位**（產品文件、新聞）→ 選 Parent Document Retrieval

**立即可行的下一步：**

1. 在現有 RAG pipeline 中加入 `SentenceWindowNodeParser`，`window_size` 從 2 開始測試
2. 用你的實際查詢測試「標準 chunk」vs「視窗擴展 chunk」傳給 LLM 的內容差異
3. 測量 token 消耗的增加量，確認在預算範圍內
4. 如果文件有明確章節結構，改用 `ParentDocumentRetriever`，父 chunk 大小設為段落邊界
5. 在 Sentence Window 基礎上疊加 cross-encoder reranker，通常能進一步提升回答品質

上下文增強不是萬靈丹，但它是在不改變底層向量索引架構的前提下，**成本最低、效果最直接的 RAG 改良手段之一**。
