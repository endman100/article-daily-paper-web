---
title: "Contextual Compression：壓縮檢索內容以降低 LLM 噪音與 Token 消耗"
description: "檢索到的文件片段往往包含大量與問題無關的資訊，直接塞給 LLM 不僅浪費 token，還會降低回答品質。Contextual Compression 在 Retriever 與 LLM 之間插入一個壓縮層，只保留真正相關的句子或段落。"
date: 2025-01-09
tags: ["RAG", "Contextual Compression", "LangChain", "Token Optimization", "LLMChainExtractor", "EmbeddingsFilter"]
---

# Contextual Compression：壓縮檢索內容以降低 LLM 噪音與 Token 消耗

標準 RAG 管道將整個 chunk 傳給 LLM，但實際上一個 512-token 的 chunk 往往只有 2-3 句話真正回答問題，其餘都是噪音。Contextual Compression 的核心思想是：在把 retrieved chunks 送往 LLM 之前，先用一個「壓縮器（Compressor）」過濾掉不相關內容，只保留與查詢高度相關的部分。這樣做同時降低了 token 消耗並提升了答案品質。

## 核心概念

Contextual Compression 解決的問題非常具體：**chunk 的粒度與查詢的粒度不匹配**。

當我們切分文件時，chunk size 是固定的（例如 512 tokens），但使用者的問題可能只需要文件中的一個數字、一個定義、或一段步驟。把整個 chunk 送進 LLM context window 有以下代價：

| 問題 | 具體影響 |
|------|----------|
| Token 浪費 | GPT-4o 每百萬 token 收費，多餘 context 直接增加成本 |
| 注意力稀釋 | LLM 在長 context 中更容易「迷失」，尤其對中間位置的資訊 |
| 延遲增加 | Context 越長，推理時間越長 |
| 噪音干擾 | 不相關段落可能讓 LLM 產生與問題無關的回答 |

壓縮的本質是：**針對特定查詢，從 chunk 中提取（Extract）或過濾（Filter）最相關的內容**。

### 三種壓縮器

LangChain 提供三種內建 Compressor，各有適用場景：

**1. LLMChainExtractor（提取型）**
- 使用 LLM 閱讀整個 chunk，提取與查詢相關的句子
- 輸出是 chunk 的子集（逐句摘取）
- 優點：精確度高，能跨句理解語義
- 缺點：每個 chunk 都需要一次 LLM 調用，成本高

**2. LLMChainFilter（過濾型）**
- 使用 LLM 判斷整個 chunk 是否相關（二元決策：保留或丟棄）
- 不修改 chunk 內容，只做取捨
- 優點：比 Extractor 便宜（prompt 更短），邏輯簡單
- 缺點：要麼全保留要麼全丟棄，粒度較粗

**3. EmbeddingsFilter（嵌入過濾型）**
- 計算 chunk embedding 與 query embedding 的餘弦相似度
- 設定閾值（如 0.76）過濾低相關度的 chunk
- 優點：無需額外 LLM 調用，速度快、成本低
- 缺點：純語義相似度，無法理解邏輯關聯

## 運作原理

標準 RAG 與 Contextual Compression RAG 的流程對比：

```
【標準 RAG 管道】
Query → Retriever → [chunk1, chunk2, chunk3] → LLM → Answer
                        ↑
               全部原始 chunk，包含大量無關內容

【Contextual Compression RAG 管道】
Query → Retriever → [chunk1, chunk2, chunk3]
                             ↓
                      Compressor（壓縮層）
                             ↓
              [compressed1, compressed2]  ← 過濾後更精簡
                             ↓
                           LLM → Answer
```

以 LLMChainExtractor 為例，壓縮流程：

```
原始 chunk（512 tokens）：
"台灣位於東亞，是一座島嶼。台灣的首都是台北，人口約 260 萬。
 台灣的 GDP 為 7600 億美元，是全球第 21 大經濟體。
 台灣以半導體產業聞名，台積電佔全球晶片代工約 55% 的市占率。
 台灣的年均溫為 22°C，夏季炎熱多颱風..."

Query：「台灣半導體產業的全球地位？」

Extractor 輸出（~60 tokens）：
"台灣以半導體產業聞名，台積電佔全球晶片代工約 55% 的市占率。"
```

Token 壓縮比：512 → 60，節省約 88%。

### ContextualCompressionRetriever 架構

```
ContextualCompressionRetriever
├── base_retriever: VectorStoreRetriever
│   └── vectorstore: ChromaDB / FAISS / etc.
└── base_compressor: DocumentCompressor
    ├── LLMChainExtractor(llm=ChatOpenAI())
    ├── LLMChainFilter(llm=ChatOpenAI())
    └── EmbeddingsFilter(embeddings=..., similarity_threshold=0.76)
```

可以透過 `DocumentCompressorPipeline` 串聯多個壓縮器（先過濾再提取）：

```
chunk → EmbeddingsFilter（快速粗篩）→ LLMChainExtractor（精確提取）→ compressed output
```

## Python 實作範例

以下是完整的 Contextual Compression RAG 實作，包含三種壓縮器的比較：

```python
import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import tiktoken

# ── 工具函數：計算 token 數量 ──────────────────────────────────────────────
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """計算文字的 token 數量"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def count_docs_tokens(docs: list) -> int:
    """計算文件列表的總 token 數量"""
    return sum(count_tokens(doc.page_content) for doc in docs)

# ── 建立測試語料庫 ────────────────────────────────────────────────────────
sample_documents = [
    Document(
        page_content="""
        台積電（TSMC）成立於 1987 年，由張忠謀創辦，總部位於新竹科學園區。
        台積電是全球最大的專業積體電路製造服務公司，佔全球晶圓代工市場約 55% 的份額。
        台積電的主要客戶包括蘋果、NVIDIA、AMD、高通等科技巨頭。
        在製程技術上，台積電已量產 3nm 製程，並持續研發 2nm 和 1.4nm 技術。
        2023 年台積電的年收入達 693 億美元，在台灣雇用超過 7 萬名員工。
        台積電在亞利桑那州、日本熊本、德國德勒斯登等地設立海外廠房，積極推動全球化布局。
        公司的護城河在於其先進的製程技術和良率控制能力，這些能力建立在數十年的工程積累上。
        """,
        metadata={"source": "tsmc_overview.txt", "chunk_id": 0}
    ),
    Document(
        page_content="""
        半導體產業鏈分為設計、製造、封裝測試三大環節。
        設計公司（Fabless）如 NVIDIA、AMD、高通，專注於晶片架構設計，不擁有自己的製造廠。
        晶圓代工廠（Foundry）如台積電、三星，提供製造服務，按照設計公司的 mask 進行生產。
        IDM（Integrated Device Manufacturer）如英特爾、三星，同時具備設計和製造能力。
        封測廠如日月光、矽品，負責晶片的封裝和測試，是最後一道工序。
        整個產業鏈高度分工，每個環節都需要專業的設備、材料和工程人才支援。
        關鍵設備供應商包括 ASML（光刻機）、應用材料（薄膜沉積）、科林研發（蝕刻）。
        """,
        metadata={"source": "semiconductor_chain.txt", "chunk_id": 1}
    ),
    Document(
        page_content="""
        Python 是一種高階、解釋型、通用程式設計語言，由 Guido van Rossum 於 1991 年發布。
        Python 以其簡潔的語法和豐富的生態系統著稱，是數據科學和機器學習領域的主流語言。
        Python 的主要特點包括：動態型別、垃圾回收、支援多種程式設計範式（物件導向、函數式、程序式）。
        Python 的標準函式庫非常豐富，涵蓋文件 I/O、網路、資料庫、正則表達式等常用功能。
        Python 3.12 引入了 PEP 695 類型別名語法和更好的錯誤訊息顯示。
        """,
        metadata={"source": "python_intro.txt", "chunk_id": 2}
    ),
    Document(
        page_content="""
        台積電的 CoWoS（Chip on Wafer on Substrate）先進封裝技術，
        是當前 AI 晶片（如 NVIDIA H100/H200）的關鍵製造工序。
        CoWoS 允許將多個晶片（CPU、GPU、HBM 記憶體）整合在同一封裝基板上，
        大幅縮短晶片間的通訊距離，降低功耗並提升頻寬。
        由於 AI 訓練對 HBM 頻寬的需求爆炸性增長，CoWoS 產能成為 AI 晶片供應鏈的瓶頸。
        台積電正積極擴充 CoWoS 產能，預計 2025 年底產能倍增。
        """,
        metadata={"source": "cowos_tech.txt", "chunk_id": 3}
    ),
]

# ── 建立向量資料庫 ─────────────────────────────────────────────────────────
print("建立向量資料庫...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(sample_documents, embeddings)

# 建立基礎 Retriever（取回 top-3）
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 方式一：LLMChainExtractor（提取相關句子）──────────────────────────────
print("\n=== 方式一：LLMChainExtractor ===")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 建立提取型壓縮器：用 LLM 從 chunk 中摘取相關句子
extractor = LLMChainExtractor.from_llm(llm)

compression_retriever_extractor = ContextualCompressionRetriever(
    base_compressor=extractor,
    base_retriever=base_retriever
)

query = "台積電在 AI 晶片供應鏈的關鍵角色是什麼？"

# 未壓縮的結果
raw_docs = base_retriever.get_relevant_documents(query)
raw_tokens = count_docs_tokens(raw_docs)
print(f"原始 chunks token 數：{raw_tokens}")

# 壓縮後的結果
start = time.time()
compressed_docs_extractor = compression_retriever_extractor.get_relevant_documents(query)
elapsed = time.time() - start

compressed_tokens = count_docs_tokens(compressed_docs_extractor)
print(f"壓縮後 token 數：{compressed_tokens}")
print(f"Token 節省：{raw_tokens - compressed_tokens}（{(1 - compressed_tokens/raw_tokens)*100:.1f}%）")
print(f"壓縮耗時：{elapsed:.2f}s")

print("\n壓縮後內容：")
for i, doc in enumerate(compressed_docs_extractor):
    print(f"[{i+1}] {doc.page_content[:200]}...")

# ── 方式二：LLMChainFilter（保留或丟棄整個 chunk）──────────────────────────
print("\n=== 方式二：LLMChainFilter ===")

# 建立過濾型壓縮器：LLM 判斷整個 chunk 是否相關
doc_filter = LLMChainFilter.from_llm(llm)

compression_retriever_filter = ContextualCompressionRetriever(
    base_compressor=doc_filter,
    base_retriever=base_retriever
)

start = time.time()
compressed_docs_filter = compression_retriever_filter.get_relevant_documents(query)
elapsed = time.time() - start

filtered_tokens = count_docs_tokens(compressed_docs_filter)
print(f"過濾後保留 {len(compressed_docs_filter)}/{len(raw_docs)} 個 chunks")
print(f"過濾後 token 數：{filtered_tokens}（節省 {(1 - filtered_tokens/raw_tokens)*100:.1f}%）")
print(f"過濾耗時：{elapsed:.2f}s")

# ── 方式三：EmbeddingsFilter（無需 LLM，使用相似度閾值）──────────────────
print("\n=== 方式三：EmbeddingsFilter ===")

# 使用 embedding 相似度過濾，閾值 0.76 表示只保留相似度 >= 0.76 的 chunk
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76  # 調高閾值 = 更嚴格過濾
)

compression_retriever_emb = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)

start = time.time()
compressed_docs_emb = compression_retriever_emb.get_relevant_documents(query)
elapsed = time.time() - start

emb_tokens = count_docs_tokens(compressed_docs_emb)
print(f"過濾後保留 {len(compressed_docs_emb)}/{len(raw_docs)} 個 chunks")
print(f"過濾後 token 數：{emb_tokens}（節省 {(1 - emb_tokens/raw_tokens)*100:.1f}%）")
print(f"過濾耗時：{elapsed:.2f}s（無 LLM 調用，速度最快）")

# ── 方式四：Pipeline 組合（先快速過濾，再精確提取）──────────────────────
print("\n=== 方式四：Pipeline（EmbeddingsFilter + LLMChainExtractor）===")

# 先用 EmbeddingsFilter 快速篩掉明顯不相關的 chunk
# 再用 LLMChainExtractor 精確提取相關句子
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.70),
        LLMChainExtractor.from_llm(llm),
    ]
)

compression_retriever_pipeline = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

start = time.time()
compressed_docs_pipeline = compression_retriever_pipeline.get_relevant_documents(query)
elapsed = time.time() - start

pipeline_tokens = count_docs_tokens(compressed_docs_pipeline)
print(f"Pipeline 壓縮後 token 數：{pipeline_tokens}（節省 {(1 - pipeline_tokens/raw_tokens)*100:.1f}%）")
print(f"Pipeline 耗時：{elapsed:.2f}s")

# ── Token 節省效果總結 ─────────────────────────────────────────────────────
print("\n=== Token 節省效果比較 ===")
print(f"{'方法':<30} {'Token 數':>10} {'節省率':>10} {'耗時':>10}")
print("-" * 65)
print(f"{'原始（無壓縮）':<30} {raw_tokens:>10} {'0%':>10} {'~0s':>10}")
print(f"{'LLMChainExtractor':<30} {compressed_tokens:>10} {f'{(1-compressed_tokens/raw_tokens)*100:.1f}%':>10}")
print(f"{'LLMChainFilter':<30} {filtered_tokens:>10} {f'{(1-filtered_tokens/raw_tokens)*100:.1f}%':>10}")
print(f"{'EmbeddingsFilter':<30} {emb_tokens:>10} {f'{(1-emb_tokens/raw_tokens)*100:.1f}%':>10}")
print(f"{'Pipeline':<30} {pipeline_tokens:>10} {f'{(1-pipeline_tokens/raw_tokens)*100:.1f}%':>10}")
```

### 整合進 RAG 問答鏈

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 自訂 Prompt 以充分利用壓縮後的精簡 context
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
根據以下上下文回答問題。上下文已經過相關性過濾，請直接利用其中的資訊作答。
若上下文中沒有足夠資訊，請明確說明。

上下文：
{context}

問題：{question}

回答："""
)

# 使用 Pipeline 壓縮器（效果最好）
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=compression_retriever_pipeline,
    chain_type_kwargs={"prompt": RAG_PROMPT},
    return_source_documents=True,
)

result = qa_chain({"query": "台積電在 AI 晶片供應鏈的關鍵角色是什麼？"})
print("回答：", result["result"])
print("\n引用來源：")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata.get('source')} ({count_tokens(doc.page_content)} tokens)")
```

## 優缺點分析

### 優點

**1. 顯著降低 Token 消耗**
在實際測試中，LLMChainExtractor 可將 context token 數壓縮 60-90%。若每日有 10,000 次查詢，每次節省 500 tokens，使用 GPT-4o 每月可節省約 $150 美元。

**2. 提升答案品質**
實驗顯示，壓縮後的 RAG 在精確性問題上準確率提升 8-15%，因為 LLM 不再被無關資訊干擾（Lost-in-the-Middle 問題緩解）。

**3. 降低延遲**
更短的 context 意味著 LLM 推理時間縮短，在低延遲要求的場景尤為明顯。

**4. 靈活的壓縮策略**
三種壓縮器可根據成本/品質需求自由選擇，或透過 Pipeline 組合使用。

### 缺點

**1. 可能截斷重要上下文**
壓縮器依賴對「相關性」的判斷，若問題需要背景知識（例如：「除了前面說的，還有什麼？」），壓縮後可能丟失關鍵資訊。

**2. LLM 壓縮器增加額外延遲和成本**
LLMChainExtractor 為每個 chunk 增加一次 LLM 調用。若檢索 5 個 chunks，則需要額外 5 次 LLM 調用（加上最終生成的 1 次），總成本可能反而上升。

**3. EmbeddingsFilter 的語義盲點**
純相似度過濾無法理解邏輯關聯。例如查詢「台積電的競爭對手是誰？」，關於「三星晶圓代工」的 chunk 語義相似度可能不夠高，但實際非常相關。

**4. 壓縮結果不確定性**
LLM 壓縮器的輸出受 prompt 和模型版本影響，相同輸入可能產生不同的壓縮結果，難以完全預測。

## 適用場景

**最適合使用 Contextual Compression 的場景：**

1. **成本敏感型應用**：新創公司、個人開發者，API 成本是重要考量
2. **長文件系統**：法律合約、學術論文、技術手冊，chunk 很長但問題很具體
3. **高頻查詢系統**：每日查詢量超過萬次，token 優化效益顯著
4. **客服知識庫**：問題通常很具體（「退款需要幾天？」），不需要整段說明

**不適合的場景：**

1. **需要完整上下文的問題**：「請總結這份報告的主要觀點」需要全文
2. **低延遲優先的場景**：LLM 壓縮器增加 RTT（往返時間），不適合 < 1s 要求
3. **多輪對話系統**：壓縮可能丟失對話歷史所需的背景資訊

## 與其他方法的比較

| 方法 | Token 消耗 | 答案精確性 | 延遲 | 實作複雜度 |
|------|-----------|-----------|------|-----------|
| 標準 RAG（無壓縮） | 高 | 中 | 低 | 低 |
| Contextual Compression（EmbeddingsFilter） | 中 | 中高 | 低 | 低 |
| Contextual Compression（LLMChainExtractor） | 低 | 高 | 高 | 中 |
| Contextual Compression（Pipeline） | 最低 | 最高 | 最高 | 中 |
| 小 Chunk Size（128 tokens） | 低 | 中（碎片化） | 低 | 低 |
| Reranking（Cross-encoder） | 高 | 高 | 中 | 中 |

**與縮小 Chunk Size 的比較**：直接切更小的 chunk 也能降低 token，但會導致語義碎片化（一個完整概念被切斷）。Contextual Compression 保留了原始 chunk 的完整語義單元，只是按需提取。

**與 Reranking 的比較**：Reranking 重新排序 chunk 但不壓縮，仍將全部或 top-k chunk 傳給 LLM；Contextual Compression 直接縮小每個 chunk 的內容，是更進一步的優化。

**兩者可以疊加使用**：先 Rerank 取最佳 top-3，再對這 3 個 chunk 做 Contextual Compression，效果通常最好。

## 小結

Contextual Compression 是 RAG 管道中一個「低成本高回報」的優化手段，核心在於解決 chunk 粒度與查詢粒度的不匹配問題。

選擇建議：
- **預算有限、需要速度**：使用 `EmbeddingsFilter`，無額外 LLM 成本
- **對精確性有較高要求**：使用 `LLMChainExtractor`
- **兩者兼顧**：使用 `DocumentCompressorPipeline` 組合

關鍵洞察：壓縮不是免費午餐。LLM 壓縮器確實增加了中間成本，但若最終生成步驟使用昂貴的 GPT-4o，而壓縮器使用 gpt-4o-mini，整體成本仍可顯著降低。實際部署時，應根據自己的查詢分佈和成本結構做 A/B 測試，而非盲目應用壓縮。
