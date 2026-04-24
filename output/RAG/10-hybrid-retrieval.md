---
title: "Hybrid Retrieval：結合向量搜尋與 BM25 的混合檢索策略"
description: "純向量搜尋對精確詞彙（人名、版本號、縮寫）天生不擅長；純 BM25 關鍵字搜尋又缺乏語義理解能力。Hybrid Retrieval 同時運行兩種搜尋，透過 RRF 或加權融合合併結果，取長補短，通常比單一方法提升 5-20% 的召回率。"
date: 2025-01-10
tags: ["RAG", "Hybrid Retrieval", "BM25", "FAISS", "RRF", "Dense Retrieval", "Sparse Retrieval"]
---

# Hybrid Retrieval：結合向量搜尋與 BM25 的混合檢索策略

向量搜尋（Dense Retrieval）和 BM25 稀疏搜尋（Sparse Retrieval）各有死角：前者在語義相似度上表現優秀，但對精確字串的匹配能力弱；後者對關鍵字精確匹配強，但不理解語義。在生產級 RAG 系統中，同時跑兩個引擎並透過 Reciprocal Rank Fusion（RRF）或加權融合合併結果，已成為提升召回率的標準做法。本文從原理到完整 Python 實作，涵蓋自製 BM25 + FAISS + RRF 混合搜尋系統。

## 核心概念

### 為什麼單一方法不夠？

**向量搜尋（Dense）的死角：精確詞彙匹配**

假設使用者查詢：「GPT-4o 的 context window 是多少 tokens？」

向量搜尋會找出語義相似的文本，例如：
- ✅「大型語言模型的上下文長度...」（語義相關）
- ❌ 但可能漏掉直接含有「GPT-4o」字串的段落，因為 embedding 將版本號「4o」視為普通字元

在以下類型的查詢中，向量搜尋容易失準：
- 版本號：`Python 3.12`、`CUDA 12.1`
- 專有名詞縮寫：`TSMC`、`RAG`、`API`
- 人名：`Jensen Huang`、`Sam Altman`
- 精確數字：`55%`、`$7.8B`

**BM25（Sparse）的死角：語義理解缺失**

BM25 基於詞頻（TF）和逆文件頻率（IDF）計算相關性，無法理解：
- 同義詞：「汽車」和「車輛」在 BM25 看來是完全不同的詞
- 語義近似：「如何加快訓練速度？」無法匹配「訓練加速技巧」
- 多語言：中英混合查詢的匹配問題

**混合檢索的互補效應：**

```
查詢：「NVIDIA H100 的 HBM3 頻寬規格」

BM25 結果：
  Rank 1: 含「H100」「HBM3」「頻寬」關鍵字的精確段落 ← 強項
  Rank 2: ...
  Rank 3: ...

向量搜尋結果：
  Rank 1: 關於 GPU 記憶體頻寬的語義相關段落
  Rank 2: 含「H100」的段落 ← 可能排在第 2
  Rank 3: ...

混合後（RRF）：
  Rank 1: 含「H100」「HBM3」且語義相關的段落 ← 兩邊都高分
```

### BM25 原理

BM25（Best Matching 25）是 TF-IDF 的改良版，公式如下：

```
Score(D, Q) = Σ IDF(qi) × [tf(qi, D) × (k1 + 1)] / [tf(qi, D) + k1 × (1 - b + b × |D|/avgdl)]

其中：
  qi   = 查詢的第 i 個詞
  tf   = 詞頻（term frequency）
  IDF  = 逆文件頻率，log((N - n(qi) + 0.5) / (n(qi) + 0.5))
  |D|  = 文件長度（詞數）
  avgdl= 所有文件的平均長度
  k1   = 詞頻飽和係數（通常 1.2~2.0）
  b    = 文件長度正規化係數（通常 0.75）
```

關鍵特性：
- **詞頻飽和**：同一詞出現 10 次不會比 3 次好 10 倍（k1 控制飽和速度）
- **文件長度懲罰**：較長文件的詞頻會被正規化（b 控制程度）
- **IDF 加權**：罕見詞（高 IDF）比常見詞更重要

### Reciprocal Rank Fusion（RRF）

RRF 是合併多個排名列表的標準算法，公式極簡：

```
RRF_Score(d) = Σ 1 / (k + rank_i(d))

其中：
  d      = 文件
  rank_i = 文件 d 在第 i 個排名列表中的位置（從 1 開始）
  k      = 平滑係數（通常設為 60）
```

**RRF 的優點：**
- 不需要對不同搜尋引擎的分數進行正規化
- 對異常高分不敏感（排名比分數更穩定）
- 只要文件在任一列表中排名高，最終得分就高

**加權融合 vs RRF 的選擇：**
- RRF：適合不確定各引擎權重時的通用選擇
- 加權融合：當你知道某種查詢應該更依賴哪個引擎時，可手動調整權重

## 運作原理

混合檢索系統的完整架構：

```
使用者查詢
    │
    ├──────────────────┬──────────────────┐
    ↓                  ↓                  ↓
BM25 稀疏搜尋    向量稠密搜尋       （可選）第三搜尋引擎
（關鍵字匹配）   （語義匹配）
    │                  │
    ↓                  ↓
[D3, D1, D5, D2]  [D1, D4, D3, D6]   ← 各自的排名列表
    │                  │
    └──────────────────┘
              ↓
        RRF / 加權融合
              ↓
    [D1, D3, D4, D2, D5, D6]          ← 合併後的排名
              ↓
         Top-K 結果
              ↓
            LLM
```

**RRF 計算範例：**

```
文件 D1：BM25 排名=2, 向量搜尋排名=1
  RRF(D1) = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252

文件 D3：BM25 排名=1, 向量搜尋排名=3
  RRF(D3) = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

文件 D4：BM25 排名=99（未找到）, 向量搜尋排名=2
  RRF(D4) = 1/(60+99) + 1/(60+2) = 0.00629 + 0.01613 = 0.02242

排序：D1 > D3 > D4 > ...
```

D1 兩邊都高排名，最終勝出。D4 只有向量搜尋找到，分數較低。

## Python 實作範例

完整的 BM25 + FAISS + RRF 混合搜尋系統：

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import defaultdict
import math

# pip install rank-bm25 faiss-cpu openai

from rank_bm25 import BM25Okapi
import faiss
from openai import OpenAI

client = OpenAI()

# ── 資料結構 ─────────────────────────────────────────────────────────────
@dataclass
class Document:
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class SearchResult:
    doc: Document
    score: float
    rank: int
    source: str  # "bm25", "dense", "hybrid"

# ── 測試語料 ─────────────────────────────────────────────────────────────
documents = [
    Document("d1", "NVIDIA H100 GPU 搭載 HBM3 記憶體，頻寬高達 3.35 TB/s，是 A100 的三倍。"),
    Document("d2", "台積電 CoWoS 先進封裝將 GPU 和 HBM 整合在同一基板，縮短通訊延遲。"),
    Document("d3", "大型語言模型（LLM）的訓練需要大量 GPU 計算資源，記憶體頻寬是瓶頸之一。"),
    Document("d4", "H100 SXM5 版本使用 NVLink 4.0 互連，提供 900 GB/s 的 GPU 間頻寬。"),
    Document("d5", "Python 機器學習框架 PyTorch 和 TensorFlow 都支援 NVIDIA CUDA 加速。"),
    Document("d6", "記憶體頻寬優化技術包括 Flash Attention、PagedAttention 等算法創新。"),
    Document("d7", "CUDA 12.1 引入了新的記憶體管理 API，改善了多 GPU 場景的效率。"),
    Document("d8", "AMD MI300X 的 HBM3 頻寬為 5.3 TB/s，在記憶體頻寬上超越 H100。"),
]

# ── BM25 稀疏搜尋器 ───────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        # 對中文文本進行字符級分詞（或使用 jieba 進行詞語分詞）
        tokenized_corpus = [self._tokenize(doc.content) for doc in documents]
        # 初始化 BM25Okapi（k1=1.5, b=0.75 為預設值）
        self.bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    def _tokenize(self, text: str) -> List[str]:
        """簡單的字符+英文詞分詞，實際應用建議使用 jieba"""
        import re
        # 分離中文字符和英文詞彙
        tokens = []
        # 提取英文/數字詞
        en_tokens = re.findall(r'[A-Za-z0-9]+(?:\.[0-9]+)?', text)
        tokens.extend([t.lower() for t in en_tokens])
        # 中文字符（字符級）
        zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(zh_chars)
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """BM25 搜尋，返回 top-k 結果"""
        query_tokens = self._tokenize(query)
        # 計算所有文件的 BM25 分數
        scores = self.bm25.get_scores(query_tokens)

        # 排序並取 top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:  # 只返回有相關性的文件
                results.append(SearchResult(
                    doc=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    source="bm25"
                ))
        return results

# ── 向量稠密搜尋器 ────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, documents: List[Document], embedding_model: str = "text-embedding-3-small"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.dimension = 1536  # text-embedding-3-small 的向量維度

        # 建立 FAISS 索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 內積（等同於歸一化後的餘弦相似度）
        self._build_index()

    def _get_embedding(self, text: str) -> np.ndarray:
        """取得文本的 embedding 向量"""
        response = client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # L2 正規化，使內積等同於餘弦相似度
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _build_index(self):
        """批次建立所有文件的 embedding 並加入 FAISS 索引"""
        print(f"建立向量索引（{len(self.documents)} 筆文件）...")
        embeddings = []
        for doc in self.documents:
            emb = self._get_embedding(doc.content)
            embeddings.append(emb)

        embeddings_matrix = np.vstack(embeddings)
        self.index.add(embeddings_matrix)
        print(f"索引建立完成，維度：{self.dimension}")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """向量相似度搜尋"""
        query_emb = self._get_embedding(query)
        query_emb = query_emb.reshape(1, -1)

        # FAISS 搜尋，返回距離和索引
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            if idx != -1 and score > 0:
                results.append(SearchResult(
                    doc=self.documents[idx],
                    score=float(score),
                    rank=rank,
                    source="dense"
                ))
        return results

# ── RRF 融合算法 ───────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60,
    top_k: int = 5
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion
    
    Args:
        result_lists: 多個搜尋引擎各自的結果列表
        k: RRF 平滑係數（通常 60）
        top_k: 最終返回的文件數量
    
    Returns:
        融合後的排名列表
    """
    # 計算每個文件的 RRF 分數
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_registry: Dict[str, Document] = {}

    for results in result_lists:
        for result in results:
            doc_id = result.doc.id
            doc_registry[doc_id] = result.doc
            # RRF 公式：1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + result.rank)

    # 排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 建立最終結果
    final_results = []
    for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
        final_results.append(SearchResult(
            doc=doc_registry[doc_id],
            score=score,
            rank=rank,
            source="hybrid"
        ))

    return final_results

# ── 加權融合算法（備選方案）──────────────────────────────────────────────
def weighted_fusion(
    bm25_results: List[SearchResult],
    dense_results: List[SearchResult],
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
    top_k: int = 5
) -> List[SearchResult]:
    """
    加權融合：對各引擎分數正規化後加權求和
    
    注意：需要先對分數進行 min-max 正規化，確保不同尺度的分數可比較
    """
    def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
        """Min-max 正規化"""
        if not results:
            return {}
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {r.doc.id: 1.0 for r in results}
        return {r.doc.id: (r.score - min_s) / (max_s - min_s) for r in results}

    bm25_norm = normalize_scores(bm25_results)
    dense_norm = normalize_scores(dense_results)

    # 合併所有文件
    all_docs = {r.doc.id: r.doc for r in bm25_results + dense_results}
    combined_scores: Dict[str, float] = {}

    for doc_id in all_docs:
        bm25_score = bm25_norm.get(doc_id, 0.0)
        dense_score = dense_norm.get(doc_id, 0.0)
        combined_scores[doc_id] = bm25_weight * bm25_score + dense_weight * dense_score

    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        SearchResult(doc=all_docs[doc_id], score=score, rank=rank, source="hybrid_weighted")
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1)
    ]

# ── 混合搜尋系統 ──────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(
        self,
        documents: List[Document],
        fusion_method: str = "rrf",  # "rrf" 或 "weighted"
        bm25_weight: float = 0.3,    # 僅加權融合時使用
        dense_weight: float = 0.7,   # 僅加權融合時使用
        rrf_k: int = 60,
    ):
        self.fusion_method = fusion_method
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        # 初始化兩個子搜尋器
        print("初始化 BM25 搜尋器...")
        self.bm25_retriever = BM25Retriever(documents)
        print("初始化向量搜尋器...")
        self.dense_retriever = DenseRetriever(documents)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """執行混合搜尋"""
        # 各自搜尋（取較多結果以確保融合後有足夠候選）
        candidate_k = max(top_k * 3, 10)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)

        # 融合
        if self.fusion_method == "rrf":
            return reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=self.rrf_k,
                top_k=top_k
            )
        else:
            return weighted_fusion(
                bm25_results, dense_results,
                self.bm25_weight, self.dense_weight,
                top_k=top_k
            )

    def search_with_comparison(self, query: str, top_k: int = 5) -> Dict:
        """搜尋並顯示各方法的比較結果"""
        candidate_k = max(top_k * 3, 10)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        hybrid_results = reciprocal_rank_fusion(
            [bm25_results, dense_results], top_k=top_k
        )

        return {
            "bm25": bm25_results[:top_k],
            "dense": dense_results[:top_k],
            "hybrid": hybrid_results
        }

# ── 主程式：展示效果對比 ──────────────────────────────────────────────────
def print_results(results: List[SearchResult], title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for r in results:
        print(f"  [{r.rank}] (score={r.score:.4f}) {r.doc.content[:60]}...")

if __name__ == "__main__":
    # 初始化混合搜尋系統
    hybrid = HybridRetriever(documents, fusion_method="rrf")

    # 測試查詢 1：精確詞彙查詢（BM25 強項）
    query1 = "H100 HBM3 頻寬"
    results1 = hybrid.search_with_comparison(query1, top_k=3)
    print(f"\n查詢：「{query1}」")
    print_results(results1["bm25"], "BM25 結果")
    print_results(results1["dense"], "向量搜尋結果")
    print_results(results1["hybrid"], "混合搜尋結果（RRF）")

    # 測試查詢 2：語義查詢（向量搜尋強項）
    query2 = "GPU 記憶體效率如何提升？"
    results2 = hybrid.search_with_comparison(query2, top_k=3)
    print(f"\n查詢：「{query2}」")
    print_results(results2["bm25"], "BM25 結果")
    print_results(results2["dense"], "向量搜尋結果")
    print_results(results2["hybrid"], "混合搜尋結果（RRF）")

    # 召回率評估（假設已知相關文件）
    print("\n=== 召回率評估 ===")
    relevant_docs = {"d1", "d4", "d8"}  # 對 query1 的標準答案

    bm25_recall = len(set(r.doc.id for r in results1["bm25"]) & relevant_docs) / len(relevant_docs)
    dense_recall = len(set(r.doc.id for r in results1["dense"]) & relevant_docs) / len(relevant_docs)
    hybrid_recall = len(set(r.doc.id for r in results1["hybrid"]) & relevant_docs) / len(relevant_docs)

    print(f"BM25 召回率：{bm25_recall:.2%}")
    print(f"向量搜尋召回率：{dense_recall:.2%}")
    print(f"混合搜尋召回率：{hybrid_recall:.2%}")
```

### 與 LangChain 整合

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever as LangchainBM25
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 建立 BM25 Retriever
bm25_retriever = LangchainBM25.from_documents(
    documents,
    k=5
)

# 建立向量 Retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# EnsembleRetriever：LangChain 內建的混合搜尋（使用 RRF）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6],  # BM25:向量 = 4:6
    c=60,                # RRF 的 k 參數
)

# 使用方式與一般 Retriever 相同
docs = ensemble_retriever.get_relevant_documents("H100 HBM3 頻寬")
```

## 優缺點分析

### 優點

**1. 互補提升召回率**
研究和實踐均顯示，混合搜尋比單一方法提升 5-20% 的召回率（Recall@K）。精確詞彙查詢（如人名、代碼）提升最明顯，有時高達 30%。

**2. 天然適合技術文檔**
程式碼搜尋（函數名、變數名）、版本號查詢、API 文件查詢等場景，都同時需要精確匹配和語義理解。

**3. RRF 簡單穩健**
RRF 不需要訓練或調參，k=60 在大多數場景下都能給出合理結果。相比複雜的 score fusion，實踐中 RRF 往往表現更好。

**4. 降低單一系統的脆弱性**
當向量模型對某類查詢失準時，BM25 可以補救；反之亦然。系統整體更穩健。

### 缺點

**1. 系統複雜度翻倍**
需要維護兩套索引（BM25 倒排索引 + 向量索引），索引更新時需同步兩者，運維成本增加。

**2. 延遲增加**
兩個搜尋引擎需要並行或串行執行（建議並行），再加上融合計算，延遲比單一搜尋高 20-50%。

**3. 儲存空間加倍**
BM25 需要儲存倒排索引，向量搜尋需要儲存 embedding 向量，語料庫越大，儲存成本越高。

**4. 權重調整需要標注數據**
加權融合的最佳 BM25/Dense 權重取決於查詢分佈，需要有代表性的標注數據集來調優。

## 適用場景

**強烈建議使用混合搜尋的場景：**

| 場景 | 原因 |
|------|------|
| 程式碼搜尋 | 函數名、變數名需要精確匹配；功能描述需要語義搜尋 |
| 法律文件 | 法條編號（第 184 條）需精確匹配；概念（過失責任）需語義搜尋 |
| 技術文檔 | 版本號、API 名稱需精確；使用方式需語義 |
| 醫療文獻 | 藥品名稱（Metformin）需精確；症狀描述需語義 |
| 企業知識庫 | 產品型號需精確；流程問題需語義 |

**可以只用向量搜尋的場景：**
- 開放式問答（語義為主）
- 長對話理解
- 個性化推薦

## 與其他方法的比較

| 方法 | 實作難度 | 召回率 | 延遲 | 精確詞彙 | 語義理解 |
|------|---------|-------|------|---------|---------|
| 純 BM25 | 低 | 中 | 最低 | ✅ 強 | ❌ 弱 |
| 純向量搜尋 | 低 | 中 | 低 | ❌ 弱 | ✅ 強 |
| 混合（RRF） | 中 | 高 | 中 | ✅ 強 | ✅ 強 |
| Reranking | 中高 | 高 | 高 | ✅ 中 | ✅ 強 |
| 混合 + Reranking | 高 | 最高 | 最高 | ✅ 強 | ✅ 最強 |

**生產級建議架構：**
```
Hybrid Retrieval（Top-20）→ Cross-Encoder Reranker（Top-5）→ LLM
```

先用混合搜尋擴大召回（高 Recall），再用 Reranker 精選（高 Precision），最後送給 LLM。

**主流向量資料庫的混合搜尋支援：**
- **Elasticsearch**：原生支援 `hybrid` 查詢，結合 BM25 和 kNN
- **Weaviate**：`hybrid` 搜尋 API，支援 alpha 參數控制 BM25/向量比重
- **Pinecone**：支援 sparse-dense hybrid，可直接上傳 BM25 sparse vector
- **Qdrant**：支援 sparse vector 和 dense vector 混合查詢

## 小結

Hybrid Retrieval 已成為生產級 RAG 系統的標準配置，核心理由是：向量搜尋和 BM25 的失誤模式（failure modes）相互補充，組合使用的召回率上限明顯高於任一單獨方法。

實作路徑建議：
1. **快速驗證**：使用 LangChain `EnsembleRetriever`，幾行代碼即可跑起來
2. **生產部署**：選擇原生支援混合搜尋的向量資料庫（Elasticsearch、Weaviate、Pinecone）
3. **進階優化**：在混合搜尋後加 Cross-Encoder Reranker，效果通常能再提升 5-10%

關鍵參數調優：RRF 的 k=60 是經驗值，一般不需要調整。若使用加權融合，建議 BM25:Dense = 3:7 作為起點，根據查詢類型分佈調整。
