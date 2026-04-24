---
title: "RSE - Relevant Span Extraction：從 Chunk 中精確提取相關片段"
description: "整個 chunk 送給 LLM 時，只有部分內容與問題相關，其餘都是噪音。RSE 透過精確提取相關片段，降低 token 使用量並提升答案精確度，是 Contextual Compression 的核心實現。"
date: "2025-01-15"
tags: ["RAG", "RSE", "Relevant Span Extraction", "Contextual Compression", "LangChain", "LLMChainExtractor", "BERT QA", "Token 優化"]
---

# RSE - Relevant Span Extraction：從 Chunk 中精確提取相關片段

RAG 系統將 chunk 全文送給 LLM，有時整份 500 token 的文件只有 2 句話和問題相關——其他 480 個 token 都是噪音。噪音不是免費的：它佔用 context window、稀釋注意力、拉高延遲、增加成本。Relevant Span Extraction（RSE）解決的就是這個問題：不是把整份 chunk 送給 LLM，而是先從 chunk 中精確提取與問題最相關的片段，再送入生成階段。

## 核心概念

**RSE（Relevant Span Extraction，相關片段提取）** 是 **Contextual Compression** 的一種具體實現：在向量搜尋找到相關文件後，進一步壓縮這些文件，只保留與當前問題直接相關的片段（span）。

「Span」在 NLP 中指的是文本中的一個連續子序列，可以是一個句子、一段話、或任意長度的片段。RSE 的目標是找到 chunk 中「回答問題所需的最小充分片段」。

**和原始 RAG 的對比**：

```
原始 RAG：
  向量搜尋 → 整份 chunk（500 token）→ LLM → 答案

RSE 增強的 RAG：
  向量搜尋 → 整份 chunk → 片段提取（只留 80 token）→ LLM → 答案
```

token 使用量減少 80%，但相關性更集中，答案品質通常不降反升。

### RSE 與 Contextual Compression 的關係

LangChain 的 `ContextualCompressionRetriever` 是 RSE 概念的框架實現：

```
ContextualCompressionRetriever
    ├── base_retriever（任何向量搜尋器）
    └── base_compressor（壓縮器，RSE 的具體實現）
            ├── LLMChainExtractor  ← 用 LLM 提取相關句子
            ├── LLMChainFilter     ← 用 LLM 過濾不相關文件
            ├── EmbeddingsFilter   ← 用嵌入相似度過濾
            └── DocumentCompressorPipeline  ← 組合多種壓縮器
```

RSE 主要對應 `LLMChainExtractor`：給定查詢和文件，讓 LLM 提取出回答查詢所需的相關句子。

## 運作原理

### 三種 RSE 實作方式

**方式一：LLM 指令提取（Prompt Engineering）**

最直接的方法：用 Prompt 指示 LLM 從文件中提取相關句子。

```
Prompt：
  給定問題：{question}
  給定文件：{document}
  
  請從文件中提取出所有與問題直接相關的句子。
  如果文件中沒有相關內容，回覆「無相關內容」。
  只輸出提取的句子，不要添加任何解釋。
```

優點：靈活，能理解語義；缺點：速度慢、成本高、可能過度提取或截斷。

**方式二：Reading Comprehension 模型（BERT QA）**

把問題當成「問題」，文件當成「篇章」，用 BERT-based QA 模型找出 answer span 的起始和結束位置。

```
輸入：[CLS] 問題 [SEP] 文件 [SEP]
輸出：start_logits, end_logits → 對每個 token 預測是否是答案的開始/結束
```

優點：速度快（本地推理），精準定位；缺點：只能提取短答案，不適合需要多句話的問題。

**方式三：混合方法（Embeddings Filter + LLM 提取）**

先用 embedding 相似度快速過濾整份文件中不相關的段落，再對保留的段落用 LLM 精細提取。這樣大幅減少 LLM 需要處理的文字量。

```
文件（500 token）
    │
    ▼ 按段落切分
[段落1] [段落2] [段落3] [段落4] [段落5]
    │
    ▼ Embedding 相似度過濾（快速）
保留：[段落2] [段落4]（與查詢相似度 > 閾值）
    │
    ▼ LLM 精細提取（只處理 200 token）
最終片段：「段落2的第3句 + 段落4的第1句」
```

### 完整流程圖

```
使用者查詢
    │
    ▼
向量搜尋（Bi-Encoder）
    │
    ▼
Top-k 候選 Chunks（每份可能 300-500 token）
    │
    ▼
┌─────────────────────────────────────┐
│        Relevant Span Extraction     │
│                                     │
│  對每份 chunk：                     │
│  1. 按句/段分割                     │
│  2. 計算每句與查詢的相關性           │
│  3. 提取相關句子（span）             │
│  4. 組合成壓縮後的 context          │
└─────────────────────────────────────┘
    │
    ▼
壓縮後的 Context（每份僅 50-100 token）
    │
    ▼
LLM 生成最終答案（Context 精準，Token 省）
```

## Python 實作範例

以下展示三種實作方式：LangChain `LLMChainExtractor`、BERT QA 模型、以及自訂 Embedding Filter。

```python
import os
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document


# ── 1. 初始化 LLM 和嵌入模型 ──

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)


# ── 2. 建立向量庫（含長文件） ──

# 刻意使用長文件，展示 RSE 的壓縮效果
long_documents = [
    """Python 安裝與環境管理完整指南

    一、安裝 Python
    訪問 python.org 下載最新版 Python 安裝程式。
    Windows 用戶安裝時務必勾選「Add Python to PATH」選項，否則命令列無法直接呼叫 python 指令。
    macOS 用戶可透過 Homebrew 安裝：brew install python3。
    Linux 用戶通常系統已預裝 Python，可用 python3 --version 確認版本。
    
    二、虛擬環境管理
    強烈建議每個專案使用獨立虛擬環境，避免套件版本衝突。
    建立虛擬環境：python -m venv venv
    啟動（Windows）：venv\\Scripts\\activate
    啟動（Unix）：source venv/bin/activate
    停用：deactivate
    
    三、套件管理
    pip install 套件名稱  # 安裝
    pip uninstall 套件名稱  # 移除
    pip list  # 列出已安裝套件
    pip freeze > requirements.txt  # 匯出相依套件清單
    pip install -r requirements.txt  # 從清單安裝""",
    
    """OpenAI API 使用指南
    
    一、取得 API 金鑰
    登入 platform.openai.com，在 API Keys 頁面建立新金鑰。
    建議將金鑰存儲為環境變數，而非硬編碼在程式中：
    export OPENAI_API_KEY="your-key-here"
    
    二、基本呼叫方式
    安裝 SDK：pip install openai
    
    from openai import OpenAI
    client = OpenAI()  # 自動讀取 OPENAI_API_KEY 環境變數
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "你好"}]
    )
    print(response.choices[0].message.content)
    
    三、速率限制與配額
    免費方案每分鐘限制 3 次 RPM（Requests Per Minute）。
    付費方案根據 Tier 不同，最高可達 10,000 RPM。
    超出限制會收到 429 Too Many Requests 錯誤。
    建議實作指數退避重試：第1次等1秒，第2次等2秒，第3次等4秒。"""
]

vectorstore = Chroma.from_texts(
    texts=long_documents,
    embedding=embeddings,
    collection_name="rse_demo"
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ── 3. 方案一：LLMChainExtractor（LLM 直接提取） ──

def demo_llm_extractor(query: str):
    """使用 LLM 從文件中提取相關片段。"""
    print(f"\n{'='*55}")
    print(f"方案一：LLMChainExtractor")
    print(f"查詢：{query}")
    
    # 建立壓縮器
    compressor = LLMChainExtractor.from_llm(llm)
    
    # 建立 ContextualCompressionRetriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # 執行壓縮檢索
    compressed_docs = compression_retriever.invoke(query)
    
    for i, doc in enumerate(compressed_docs):
        original_len = len(doc.metadata.get("_source", doc.page_content))
        compressed_len = len(doc.page_content)
        print(f"\n文件 {i+1} 壓縮結果（{compressed_len} 字）：")
        print(f"{doc.page_content}")
    
    return compressed_docs


# ── 4. 方案二：EmbeddingsFilter（Embedding 快速過濾） ──

def demo_embeddings_filter(query: str, similarity_threshold: float = 0.76):
    """
    用 Embedding 相似度快速過濾不相關句子。
    比 LLMChainExtractor 快 10 倍以上，但精準度略低。
    """
    print(f"\n{'='*55}")
    print(f"方案二：EmbeddingsFilter（閾值={similarity_threshold}）")
    print(f"查詢：{query}")
    
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )
    
    filtered_docs = compression_retriever.invoke(query)
    
    for i, doc in enumerate(filtered_docs):
        print(f"\n過濾後文件 {i+1}（{len(doc.page_content)} 字）：")
        print(f"{doc.page_content[:200]}...")
    
    return filtered_docs


# ── 5. 方案三：Pipeline（組合壓縮器，效果最佳） ──

def demo_pipeline_compressor(query: str):
    """
    組合壓縮器管道：先用 Embedding 過濾，再去重，最後 LLM 精提取。
    兼顧速度和精準度。
    """
    print(f"\n{'='*55}")
    print(f"方案三：Pipeline 組合壓縮器")
    print(f"查詢：{query}")
    
    # 步驟1：EmbeddingsFilter 快速粗篩
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.72
    )
    
    # 步驟2：去除語義重複的文件段落
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    
    # 步驟3：LLM 精確提取最相關句子
    llm_extractor = LLMChainExtractor.from_llm(llm)
    
    # 組合管道
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[embeddings_filter, redundant_filter, llm_extractor]
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )
    
    results = compression_retriever.invoke(query)
    
    for i, doc in enumerate(results):
        print(f"\n管道壓縮結果 {i+1}（{len(doc.page_content)} 字）：")
        print(f"{doc.page_content}")
    
    return results


# ── 6. BERT QA 模型（Reading Comprehension 方式） ──

def bert_qa_extraction(question: str, context: str) -> dict:
    """
    使用 BERT-based QA 模型從文件中精確定位答案片段。
    適合需要精確短答案的場景（如數字、名稱、日期）。
    """
    from transformers import pipeline
    
    # 使用 multilingual BERT QA 模型（支援中文）
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/bert-base-cased-squad2",
        # 中文場景建議：hfl/chinese-macbert-base
    )
    
    result = qa_pipeline(
        question=question,
        context=context,
        max_answer_len=100,
        handle_impossible_answer=True  # 若無答案，回傳空字串而非強行提取
    )
    
    print(f"\nBERT QA 提取結果：")
    print(f"  答案：{result['answer']}")
    print(f"  信心分數：{result['score']:.4f}")
    print(f"  位置：第 {result['start']} 至 {result['end']} 字符")
    
    return result


# ── 7. Token 使用量比較 ──

def compare_token_usage(query: str, documents: List[str]):
    """
    計算並比較原始方式 vs RSE 方式的 token 使用量。
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    
    # 方式一：原始（整份 chunk）
    full_tokens = sum(len(enc.encode(doc)) for doc in documents)
    
    # 方式二：假設 RSE 提取後只保留 20% 的內容（保守估計）
    estimated_compressed = int(full_tokens * 0.2)
    
    print(f"\nToken 使用量比較（查詢：{query[:30]}）：")
    print(f"  原始 chunk 總計：{full_tokens:,} tokens")
    print(f"  RSE 壓縮後（估計）：{estimated_compressed:,} tokens")
    print(f"  節省：{full_tokens - estimated_compressed:,} tokens（{(1 - estimated_compressed/full_tokens)*100:.0f}%）")
    
    # 換算成 GPT-4o-mini 費用
    input_cost_per_1m = 0.15  # USD per 1M input tokens
    saving_usd = (full_tokens - estimated_compressed) / 1_000_000 * input_cost_per_1m
    print(f"  每次查詢節省約 ${saving_usd:.6f}（@$0.15/1M tokens）")


# ── 使用範例 ──

if __name__ == "__main__":
    test_query = "如何設定 API 速率限制的重試機制？"
    
    # 展示各種 RSE 方式
    demo_llm_extractor(test_query)
    demo_embeddings_filter(test_query)
    demo_pipeline_compressor(test_query)
    
    # Token 使用量分析
    compare_token_usage(test_query, long_documents)
```

**安裝依賴**：
```bash
pip install langchain langchain-openai langchain-chroma chromadb tiktoken
# 若使用 BERT QA：
pip install transformers torch
```

## 優缺點分析

### 優點

**1. 大幅降低 Token 使用量**：典型場景下，RSE 可以將送給 LLM 的 context 壓縮至原來的 15-30%，直接降低 API 成本和推理延遲。

**2. 減少 LLM 的注意力稀釋**：研究顯示 LLM 對 context 中的噪音內容會分散注意力（Lost in the Middle 問題）。RSE 讓 LLM 只看到和問題直接相關的文字。

**3. 支援精確引用**：提取的片段可以直接作為答案的出處引用，不像整份 chunk 那樣模糊。

**4. 對長文件特別有效**：文件越長，RSE 的相對收益越大。對於學術論文、法律文書、技術手冊等長文檔，RSE 幾乎是必備的。

### 缺點

**1. 可能過度截斷上下文**：有些問題需要結合前後文才能正確理解，RSE 如果提取範圍過小，可能導致歧義甚至錯誤。

**2. LLMChainExtractor 增加額外 LLM 呼叫**：每份候選文件都需要一次 LLM 推理，若第一階段返回 10 份文件，就需要 10 次額外呼叫，延遲顯著增加。

**3. EmbeddingsFilter 精準度受限**：基於 Embedding 相似度的過濾無法理解語義轉折（例如「雖然A，但實際上B」，查詢關心B，卻可能因為A的相似度較高而被保留）。

**4. 短 chunk 效果有限**：如果 chunk 本來就只有 100-150 token，RSE 的壓縮空間很小，不值得增加這一層的複雜度。

## 適用場景

**長文件問答（技術文件、學術論文）**：文件動輒 2000-5000 token，RAG 取到的 chunk 仍然很長。RSE 能精確找到和問題相關的 2-3 句話，大幅降低 LLM 需要處理的資訊量。

**需要精確引用的場景（法律、醫療）**：法律文件需要標注「具體哪一條哪一款」，醫療知識庫需要「原文表述是什麼」。RSE 提取的片段可以直接作為引用根據。

**Context Window 緊張的場景**：當多份文件的 context 加起來接近模型的 context window 上限時，RSE 可以有效「騰出空間」讓更多文件進來。

**成本敏感的高流量系統**：每次查詢節省 70-80% 的 input token，在百萬級日活系統中，每月可節省數千美元的 API 費用。

**對比文件差異分析**：需要比較多份文件的特定條款時，RSE 先提取各份文件的相關句子，再讓 LLM 做比較，比把所有文件全文送入高效得多。

## 與其他方法的比較

| 方法 | 目標 | 操作層級 | 對 Token 的影響 | 對精準度的影響 |
|------|------|---------|---------------|--------------|
| **RSE** | 壓縮 context | Chunk 內部片段 | ↓ 70-80% | ↑ 精準 |
| **Chunking 優化** | 改善切分 | 文件切分策略 | 中性 | ↑ 連貫性 |
| **Reranker** | 重排序 | Chunk 排序 | 不變 | ↑ 相關性 |
| **Multi-Query** | 擴大召回 | 查詢生成 | ↑（多次搜尋） | ↑ 召回率 |

**RSE vs. Reranker 的協作關係**：這兩個方法解決不同層面的問題，可以疊加使用：Reranker 決定「哪些 chunk 排在前面」，RSE 決定「這個 chunk 裡哪些句子送給 LLM」。先 Reranker 選出最相關的 5 份文件，再 RSE 從每份文件中提取最相關的句子，是目前效果最好的 RAG 精排架構之一。

## 小結

RSE 是 RAG 優化鏈條中常被忽略但 ROI 極高的一環。多數系統在實作 RAG 時會花大量精力在 chunking 策略和向量搜尋調優，卻忽略了「搜尋到的 chunk 裡可能只有 20% 的內容有用」這個問題。

實作建議：

1. **先量化問題規模**：用 tiktoken 計算現有系統每次查詢的平均 context token 數。若超過 1500 token，RSE 就有顯著收益。

2. **優先試 EmbeddingsFilter**：它無需額外 LLM 呼叫，延遲影響極小，可以快速評估過濾效果。

3. **LLMChainExtractor 用於精準度要求高的場景**：每份文件都讓 LLM 判斷哪些句子相關，精準度最高，但會增加 1-2 秒延遲。

4. **注意 chunk 大小下限**：建議 RSE 只對超過 300 token 的 chunk 執行，太短的 chunk 直接全文傳遞即可。

5. **Pipeline 方案是生產首選**：EmbeddingsFilter 粗篩 → 去重過濾 → LLMChainExtractor 精提取，三層組合在精準度和速度之間取得最佳平衡。
