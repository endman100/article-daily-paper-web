---
title: "Hierarchical Indices：Small-to-Big Retrieval 解決 Chunk 困境"
description: "小 chunk 搜尋精確但缺乏上下文；大 chunk 上下文豐富但向量表示不精確。Hierarchical Indices 建立兩層索引結構，用小 chunk 定位、大 chunk 回答，兼得精確性與完整性。"
date: 2024-01-15
tags: ["RAG", "Hierarchical Indices", "Small-to-Big", "LlamaIndex", "Chunk Strategy", "AutoMerging"]
---

# Hierarchical Indices：Small-to-Big Retrieval 解決 Chunk 困境

Hierarchical Indices（層次索引）是解決 RAG 系統中長久存在的「Chunk 兩難」的優雅方案。在 LlamaIndex 社群中，它也被稱為 Small-to-Big Retrieval 或 Auto-Merging Retrieval。核心思想極為直觀：用小片段精確搜尋、用大片段完整回答，通過父子節點關係將兩個層次串聯。這個方案在長文件問答場景中，相比單一層次 RAG 有顯著的效果提升。

---

## 核心概念

### Chunk 大小的兩難困境

在設計 RAG 系統時，Chunk 大小是最重要也最難抉擇的超參數之一：

**小 Chunk（128-256 tokens）的問題**
```
優點：向量表示聚焦，搜尋精確度高
缺點：缺乏上下文 → LLM 看到片段，無法理解全局
      語義不完整 → 一個概念可能被切斷在兩個 chunk 之間
```

**大 Chunk（1024-2048 tokens）的問題**
```
優點：包含完整上下文，LLM 理解能力強
缺點：向量空間中「主題漂移」→ 一個 chunk 包含多個主題
      搜尋精確度低 → 向量表示被多個主題稀釋
```

這是一個真實的矛盾：嵌入模型在短文本上表現最好，LLM 在長上下文上理解最好。

### Hierarchical Indices 的解法

層次索引的核心洞察是：**搜尋和生成是兩個不同的任務，可以使用不同粒度的 chunk**。

```
索引結構（兩層）：

第一層（Summary Layer）
┌─────────────────────────────────────┐
│ 文件摘要 / 段落摘要                  │
│ 約 512-1024 tokens                  │
│ 用於：粗粒度篩選，判斷文件是否相關   │
└─────────────────────────────────────┘
            ↕ 父子關係
第二層（Chunk Layer）
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│chunk1│ │chunk2│ │chunk3│ │chunk4│
│128t  │ │128t  │ │128t  │ │128t  │
└──────┘ └──────┘ └──────┘ └──────┘
  小 chunk，精確搜尋，指向父節點
```

---

## 運作原理

### 三層架構詳解

**Layer 1：文件摘要層（Document Summary）**
- 每個原始文件對應一個摘要節點
- 可以是 LLM 生成的摘要，也可以是文件的前幾段
- 用於判斷「這個文件是否可能包含答案」

**Layer 2：父 Chunk 層（Parent Chunks）**
- 文件被分割成較大的 chunk（512-1024 tokens）
- 每個父 chunk 包含完整的段落或邏輯單元
- 這是最終送給 LLM 的上下文

**Layer 3：子 Chunk 層（Child Chunks）**
- 父 chunk 進一步分割成小 chunk（128-256 tokens）
- 這是向量資料庫中實際索引的單元
- 搜尋匹配時使用這一層

### Small-to-Big 完整流程

```
用戶查詢
    │
    ▼
┌──────────────────────────────┐
│  向量搜尋子 Chunk 層          │
│  找到最相關的小 chunk（精確）  │
└──────────────────────────────┘
    │
    ▼ 每個命中的子 chunk
┌──────────────────────────────┐
│  追溯父節點                   │
│  從子 chunk 獲取對應父 chunk  │
└──────────────────────────────┘
    │
    ▼ 合併：如果同一父節點的子 chunk 命中 ≥ N 個
┌──────────────────────────────┐
│  Auto-Merging（自動合併）     │
│  用父 chunk 替換多個子 chunk  │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  將父 chunk（完整上下文）      │
│  送入 LLM 生成最終答案        │
└──────────────────────────────┘
```

### Auto-Merging 邏輯

Auto-Merging 是層次索引的關鍵機制：

```
假設父 chunk P 有 4 個子 chunk: C1, C2, C3, C4
設定合併閾值 threshold = 0.5（50%的子節點命中）

情況 A：命中 C1, C3（2個 = 50%）
→ 觸發合併，返回整個父 chunk P（更完整的上下文）

情況 B：命中 C1（1個 = 25%）
→ 不觸發合併，只返回 C1

邏輯：如果一個父 chunk 的大部分子節點都被認為相關
      說明這個父 chunk 整體都是相關的，直接用父節點更好
```

---

## Python 實作範例

### 使用 LlamaIndex 的 HierarchicalNodeParser + AutoMergingRetriever

```python
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os

# ─────────────────────────────────────
# 步驟 1：設定全域設定
# ─────────────────────────────────────
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ─────────────────────────────────────
# 步驟 2：準備文件
# ─────────────────────────────────────
# 模擬一份長篇技術文件（實際使用可從 PDF/網頁讀取）
long_document_text = """
# 台積電技術發展史

## 第一章：公司起源與早期發展

台積電（Taiwan Semiconductor Manufacturing Company, TSMC）由張忠謀於1987年在台灣新竹科學園區創立。
這家公司開創了「晶圓代工」（Foundry）的全新商業模式，徹底改變了半導體產業的生態系統。

在創立之前，張忠謀曾在德州儀器（Texas Instruments）工作長達25年，擔任全球半導體業務副總裁。
他觀察到當時的半導體產業有一個巨大的市場缺口：許多 IC 設計公司有傑出的設計能力，
卻苦於沒有足夠的資本建立自己的製造廠。台積電的成立正是為了填補這個缺口。

## 第二章：製程技術的演進

### 90nm 到 28nm 時代（2003-2011）

進入21世紀，台積電的製程技術不斷突破。2003年量產90nm製程，2005年轉入65nm，
2007年推出45nm，2011年成功量產28nm。在28nm製程時代，台積電確立了其在代工市場的領導地位。

### 16nm FinFET 的轉折點（2015）

2015年，台積電成功量產16nm FinFET製程，這是一個重要的轉折點。
FinFET（鰭式場效電晶體）架構解決了傳統平面電晶體在10nm以下遭遇的量子穿隧問題，
使得摩爾定律得以繼續延伸。Apple A9 處理器是第一個採用台積電16nm FinFET製程的旗艦產品。

### 7nm 時代：與三星的競爭白熱化（2018）

2018年，台積電量產7nm製程，比三星早約半年進入量產。Apple A12 Bionic 是全球第一款
採用7nm製程的商用晶片，搭載在iPhone XS系列中。7nm製程使晶片效能提升約20%，
功耗降低約40%，相比10nm製程。

### 5nm 與 3nm：拉開差距（2020-2022）

台積電在2020年量產5nm，2022年量產3nm（N3），進一步鞏固其技術領先地位。
3nm製程採用了改良的FinFET架構，相比5nm製程邏輯密度提升約60%，效能提升約15%，
功耗降低約30%。Apple M3 Pro 和 M3 Max 處理器均採用此製程。

## 第三章：主要客戶與產品應用

### Apple：最大也是最緊密的夥伴

Apple 是台積電最重要的客戶，佔台積電年營收約25-30%。雙方的合作始於2013年，
Apple 將 A7 晶片的部分訂單交給台積電試產。2016年起，Apple 將所有 A 系列晶片
（iPhone 處理器）和後來的 M 系列晶片（Mac 處理器）獨家委托台積電製造。

主要產品包括：
- A 系列：iPhone、iPad 的主處理器
- M 系列：MacBook、Mac Mini、Mac Studio 的處理器
- W 系列：AirPods 的處理器
- T 系列：Mac 安全晶片

### NVIDIA：AI 時代的新主角

隨著 AI 需求爆發，NVIDIA 對台積電的重要性快速提升。NVIDIA 的 H100 GPU
採用台積電4nm製程，A100 採用7nm製程。2023年起，NVIDIA 的 AI 晶片訂單
成為台積電成長的最重要驅動力。台積電的 CoWoS 先進封裝技術也因此需求
出現嚴重供不應求的情況。

## 第四章：先進封裝技術

### CoWoS（Chip-on-Wafer-on-Substrate）

CoWoS 是台積電的旗艦先進封裝技術，允許將多個晶片（Die）整合在同一個封裝基板上，
透過矽中介層（Silicon Interposer）高密度連接，大幅提升記憶體頻寬和降低延遲。

NVIDIA H100 GPU 使用 CoWoS 將 GPU Die 與 HBM3 記憶體整合，實現3.35 TB/s
的超高記憶體頻寬，這是 AI 訓練任務所必需的。

### SoIC（System-on-Integrated-Chips）

SoIC 是更先進的3D堆疊封裝技術，允許晶片垂直堆疊，進一步縮短晶片間的通訊距離。
台積電預計 SoIC 技術將在2nm製程時代被廣泛採用。
"""

documents = [Document(text=long_document_text)]

# ─────────────────────────────────────
# 步驟 3：建立層次節點解析器
# ─────────────────────────────────────
# 定義三層 chunk 大小：文件層 → 父層 → 子層
# 搜尋時使用最小的葉節點（子層），回答時使用父層
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 由大到小：文件層 → 父層 → 子層
)

# 解析文件，建立層次節點樹
nodes = node_parser.get_nodes_from_documents(documents)

# 取出葉節點（最小的子 chunk，用於向量索引）
leaf_nodes = get_leaf_nodes(nodes)

print(f"總節點數：{len(nodes)}")
print(f"葉節點數（用於搜尋）：{len(leaf_nodes)}")

# ─────────────────────────────────────
# 步驟 4：建立儲存和索引
# ─────────────────────────────────────
# 所有節點（含父節點）存入 docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

# 只對葉節點（小 chunk）建立向量索引
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)

# ─────────────────────────────────────
# 步驟 5：建立 AutoMergingRetriever
# ─────────────────────────────────────
# base_retriever：在葉節點中搜尋
base_retriever = base_index.as_retriever(similarity_top_k=6)

# AutoMergingRetriever：根據命中情況決定是否合併到父節點
retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True,      # 顯示合併決策過程
    simple_ratio_thresh=0.5  # 50%子節點命中時觸發合併
)

# ─────────────────────────────────────
# 步驟 6：建立查詢引擎
# ─────────────────────────────────────
query_engine = RetrieverQueryEngine.from_args(retriever)

# ─────────────────────────────────────
# 步驟 7：對比測試（有無層次索引）
# ─────────────────────────────────────
def compare_retrieval(question: str):
    """對比層次索引和普通索引的效果"""
    print(f"\n{'='*60}")
    print(f"問題：{question}")
    print('='*60)
    
    # 方法 A：普通向量索引（只用小 chunk）
    plain_index = VectorStoreIndex(leaf_nodes)
    plain_engine = plain_index.as_query_engine(similarity_top_k=3)
    plain_result = plain_engine.query(question)
    
    print("\n【普通 RAG（小 chunk）答案】：")
    print(plain_result.response)
    print(f"\n使用的 chunk 數：{len(plain_result.source_nodes)}")
    for node in plain_result.source_nodes:
        print(f"  - {node.text[:100]}...")
    
    # 方法 B：層次索引（Small-to-Big）
    hier_result = query_engine.query(question)
    
    print("\n【層次索引（Small-to-Big）答案】：")
    print(hier_result.response)
    print(f"\n使用的節點數：{len(hier_result.source_nodes)}")
    for node in hier_result.source_nodes:
        print(f"  - 長度: {len(node.text)} tokens | {node.text[:100]}...")

# 測試問題
compare_retrieval("台積電的 3nm 製程有哪些技術特點和應用？")
compare_retrieval("CoWoS 封裝技術如何幫助 NVIDIA 的 AI 晶片？")
compare_retrieval("台積電和 Apple 的合作歷程是什麼？")
```

### 自訂兩層索引（不依賴 LlamaIndex）

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

# ─────────────────────────────────────
# 手動實作 Parent-Child 層次索引
# ─────────────────────────────────────

def build_hierarchical_index(text: str):
    """建立父子層次索引"""
    embeddings = OpenAIEmbeddings()
    
    # 分割父 chunk（大，用於回答）
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    parent_chunks = parent_splitter.create_documents([text])
    
    # 為每個父 chunk 分配 ID
    for i, chunk in enumerate(parent_chunks):
        chunk.metadata["chunk_id"] = f"parent_{i}"
        chunk.metadata["is_parent"] = True
    
    # 分割子 chunk（小，用於搜尋）
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    
    child_chunks = []
    # 為每個父 chunk 創建對應的子 chunk
    for parent in parent_chunks:
        children = child_splitter.create_documents([parent.page_content])
        for child in children:
            child.metadata["parent_chunk_id"] = parent.metadata["chunk_id"]
            child.metadata["is_parent"] = False
        child_chunks.extend(children)
    
    print(f"父 chunk 數：{len(parent_chunks)}")
    print(f"子 chunk 數：{len(child_chunks)}")
    
    # 只對子 chunk 建立向量索引
    child_vectorstore = Chroma.from_documents(
        child_chunks,
        embeddings,
        collection_name="child_chunks"
    )
    
    # 父 chunk 存在字典中供查詢
    parent_store = {
        chunk.metadata["chunk_id"]: chunk
        for chunk in parent_chunks
    }
    
    return child_vectorstore, parent_store


def small_to_big_retrieve(query: str, child_vectorstore, parent_store, k=4):
    """Small-to-Big 檢索：搜尋小 chunk，返回對應父 chunk"""
    # 搜尋子 chunk（小而精確）
    child_results = child_vectorstore.similarity_search(query, k=k)
    
    # 統計每個父 chunk 被命中的次數
    parent_hit_count = {}
    for child in child_results:
        parent_id = child.metadata.get("parent_chunk_id")
        if parent_id:
            parent_hit_count[parent_id] = parent_hit_count.get(parent_id, 0) + 1
    
    print(f"\n子 chunk 搜尋結果：{len(child_results)} 個")
    print(f"涉及的父 chunk：{parent_hit_count}")
    
    # 返回被命中的父 chunk（完整上下文）
    retrieved_parents = []
    for parent_id in parent_hit_count:
        if parent_id in parent_store:
            retrieved_parents.append(parent_store[parent_id])
    
    return retrieved_parents


# 測試
child_vs, parent_store = build_hierarchical_index(long_document_text)
parents = small_to_big_retrieve(
    "台積電的 CoWoS 技術和 NVIDIA 的關係",
    child_vs,
    parent_store
)

llm = ChatOpenAI(model="gpt-4o")
context = "\n\n".join([p.page_content for p in parents])
answer = llm.invoke(f"根據以下內容回答問題：\n{context}\n\n問題：台積電的 CoWoS 技術和 NVIDIA 的關係")
print(answer.content)
```

---

## 優缺點分析

### 優點

**1. 兼得精確搜尋與完整上下文**
這是最直接的優點。小 chunk 的精確搜尋 + 大 chunk 的完整上下文，理論上比任何固定大小的 chunk 策略都更好。

**2. 自然的文件結構保留**
父子關係可以對應文件的自然結構（章節→段落→句子），使分割更符合語義邏輯。

**3. 降低幻覺風險**
LLM 獲得更完整的上下文，不需要「猜測」被切斷的部分，減少因上下文不足引起的幻覺。

**4. 在長文件問答上效果顯著**
研究和實際測試表明，在需要理解長文件結構的問題上，層次索引比扁平索引有明顯優勢。

### 缺點

**1. 儲存開銷增加**
同一份內容在不同層次都被儲存，空間消耗約為單層的 1.5-2 倍。

**2. 索引建立時間更長**
需要多次分割文件並建立多個層次的關係，初始化成本較高。

**3. 有時父 chunk 包含不必要的內容**
當子 chunk 命中觸發合併後，整個父 chunk 被返回，可能包含與問題無關的段落，增加 context 長度。

**4. 閾值設定需要調優**
Auto-Merging 的觸發閾值（如 50%）需要根據具體應用場景調整，沒有通用最佳值。

---

## 適用場景

### 最適合的場景

**書籍和長報告問答**
一本書有章節、小節、段落的自然層次，Hierarchical Indices 完美對應這種結構。問「第三章的核心論點」和問「某個具體數據」可以分別在不同層次找到最佳答案。

**技術手冊和 API 文件**
技術文件通常有嚴格的層次結構，且用戶既可能問宏觀問題（「這個功能是做什麼的？」）也可能問具體問題（「這個參數的默認值是什麼？」）。

**法律文件分析**
合約和法規有條款→子條款的天然層次，且需要在精確定位具體條款的同時理解其完整的法律背景。

**學術論文集**
需要在論文層級做整體篩選，在段落層級做精確引用。

### 不太適合的場景

- 短文本庫（每篇文件本身就很短，沒有分層的必要）
- 非結構化、流式文本（沒有清晰的章節層次）
- 對索引建立速度要求極高的場景

---

## 與其他方法的比較

| 特性 | 固定小 Chunk | 固定大 Chunk | Hierarchical Indices |
|------|------------|------------|---------------------|
| 搜尋精確度 | ✓✓✓ | ✓ | ✓✓✓ |
| 上下文完整性 | ✗ | ✓✓✓ | ✓✓✓ |
| 儲存效率 | 高 | 高 | 中（多層儲存）|
| 建立複雜度 | 低 | 低 | 中 |
| 長文件問答 | 差 | 中 | 優秀 |

**vs. Sentence Window Retrieval**：
另一種類似策略是 Sentence Window——以句子為單位搜尋，但返回句子周圍的窗口文本。兩者思路相近，但 Hierarchical Indices 的分層結構更靈活，可以有三層甚至更多層，而 Sentence Window 只有兩層（句子 + 窗口）。

**vs. Parent Document Retriever（LangChain）**：
LangChain 的 ParentDocumentRetriever 實現了類似功能，邏輯幾乎相同。LlamaIndex 的 AutoMergingRetriever 增加了「合併閾值」的智能判斷，而 LangChain 版本默認總是返回父文件。

---

## 小結

Hierarchical Indices 以一種優雅的方式解決了 RAG 中最基本的工程問題：Chunk 大小的兩難。通過建立小→大的層次結構，讓搜尋和生成各司其職，使用最適合自己的粒度。

這個技術的實施並不複雜，LlamaIndex 的 `HierarchicalNodeParser` + `AutoMergingRetriever` 只需幾行代碼就能啟用。對於處理書籍、長報告、技術手冊等長文件的 RAG 應用，它幾乎是必須採用的優化策略。

在生產環境中，建議將 Hierarchical Indices 與其他優化技術（如 HyDE 查詢擴展、Reranker 重排序）結合使用，構建更完整的高品質 RAG 管道。
