---
title: "Reranker：向量搜尋之後的精準二次排序"
description: "向量搜尋的語義相似度不等於答案相關性，Top-k 結果裡可能混入大量噪音。Cross-Encoder Reranker 透過二次精排，顯著提升 RAG 的最終回答品質。"
date: "2025-01-15"
tags: ["RAG", "Reranker", "Cross-Encoder", "Bi-Encoder", "Cohere Rerank", "BGE-Reranker", "ColBERT", "NDCG", "語義搜尋"]
---

# Reranker：向量搜尋之後的精準二次排序

向量搜尋找到了 10 份文件，但最相關的那份排在第 7 位，LLM 在前幾份噪音的干擾下給出了錯誤答案——這是 RAG 系統中一個非常具體的問題。Reranker 解決的就是這個問題：把「找得到」和「排得準」分成兩個獨立的階段，用更精準但較慢的 Cross-Encoder 模型對第一階段的結果進行二次精排。

## 核心概念

**Reranker（重排序器）** 是一種在向量搜尋後執行的精排模型，核心是 **Cross-Encoder** 架構：

- **Bi-Encoder（向量搜尋用）**：查詢和文件各自獨立編碼，計算 cosine 相似度。速度快，但因為兩邊不互動，精準度受限。
- **Cross-Encoder（Reranker 用）**：查詢和文件**拼接**在一起輸入模型，讓 Transformer 的 attention 機制在兩者之間充分交互，輸出一個 0-1 的相關性分數。精準度更高，但每次都要重新計算，無法預先建立索引。

這是一個速度與精準度的典型 trade-off：向量搜尋用快速的 Bi-Encoder 篩出候選集，Reranker 用精準的 Cross-Encoder 從候選集中找出最相關的文件。

## 運作原理

### 兩階段架構（Bi-Encoder + Cross-Encoder）

```
使用者查詢
    │
    ▼
第一階段：向量搜尋（Bi-Encoder）
─────────────────────────────────
 查詢 Embedding ──→ 向量資料庫
                    （ANN 近似搜尋）
                    Top-100 候選文件
─────────────────────────────────
    │  速度快（毫秒級）
    │  精準度中等（有噪音）
    ▼

第二階段：重排序（Cross-Encoder Reranker）
─────────────────────────────────────────
 [查詢 + 文件1] → Cross-Encoder → 分數 0.92
 [查詢 + 文件2] → Cross-Encoder → 分數 0.45
 [查詢 + 文件3] → Cross-Encoder → 分數 0.87
 ...（對每份候選文件重複）
─────────────────────────────────────────
    │  速度較慢（需對每份文件推理一次）
    │  精準度高（充分理解查詢-文件關係）
    ▼

按 Cross-Encoder 分數重新排序
    │
    ▼
Top-n 文件（通常 n << k，如取前 3-5 份）
    │
    ▼
送入 LLM 生成最終答案
```

**典型參數設置**：第一階段取 Top-50 或 Top-100，Reranker 精排後取 Top-3 或 Top-5 送給 LLM。這樣在召回率（第一階段寬一點）和精準率（第二階段嚴一點）之間取得平衡。

### Bi-Encoder vs. Cross-Encoder 深度對比

| 特性 | Bi-Encoder | Cross-Encoder |
|------|------------|---------------|
| 查詢-文件互動 | 無（各自獨立編碼） | 有（拼接後聯合推理） |
| 向量預計算 | ✓ 可預計算文件 Embedding | ✗ 每次需重新計算 |
| 推理速度 | 毫秒（ANN 搜尋） | 線性 O(n)，n 為候選數量 |
| 精準度 | 中等 | 高 |
| 適合場景 | 大規模初篩 | 小批量精排 |

### 主流 Reranker 模型選型

| 模型 | 類型 | 特點 | 適用場景 |
|------|------|------|---------|
| **Cohere Rerank** | 雲端 API | 免部署，效果穩定，支援多語言 | 快速原型、生產環境 |
| **BGE-Reranker** | 開源本地 | BAAI 出品，中英文效果強 | 中文知識庫、自建部署 |
| **ms-marco-MiniLM** | 開源本地 | 輕量，推理速度快 | 延遲敏感場景 |
| **Flashrank** | 開源本地 | 極輕量，CPU 可跑 | 邊緣部署、低成本 |
| **ColBERT** | 特殊架構 | Token 級別交互，兼顧速度與精準 | 高精準度需求 |

**ColBERT 補充說明**：ColBERT 是介於 Bi-Encoder 和 Cross-Encoder 之間的架構，對每個 token 生成向量，用 MaxSim 操作計算相似度。比 Cross-Encoder 快，比 Bi-Encoder 精準，但索引體積大很多。

## Python 實作範例

以下示範兩種方案：本地 `sentence-transformers` CrossEncoder，以及 Cohere Rerank API。

```python
import os
from typing import List, Tuple
import numpy as np

# ── 方案一：sentence-transformers CrossEncoder（本地）──

from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import faiss


class TwoStageRetriever:
    """
    兩階段檢索器：Bi-Encoder 初篩 + Cross-Encoder 精排。
    """
    
    def __init__(
        self,
        bi_encoder_model: str = "BAAI/bge-small-zh-v1.5",
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        first_stage_k: int = 50,
        final_top_n: int = 5
    ):
        print(f"載入 Bi-Encoder: {bi_encoder_model}")
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        
        print(f"載入 Cross-Encoder: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        self.first_stage_k = first_stage_k
        self.final_top_n = final_top_n
        
        # FAISS 索引（第一階段用）
        self.index = None
        self.documents = []
    
    def build_index(self, documents: List[str]):
        """建立 FAISS 向量索引。"""
        self.documents = documents
        print(f"為 {len(documents)} 份文件建立 Embedding...")
        
        # 批量 Embedding
        embeddings = self.bi_encoder.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # 正規化後可用內積代替 cosine
        )
        
        # 建立 FAISS IndexFlatIP（內積索引）
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"索引建立完成，向量維度: {dim}")
    
    def retrieve(self, query: str) -> List[Tuple[str, float, int]]:
        """
        兩階段檢索：
        1. Bi-Encoder 找出 Top-k 候選
        2. Cross-Encoder 精排，返回 Top-n
        
        Returns:
            [(文件內容, rerank分數, 原始排名), ...]
        """
        if self.index is None:
            raise ValueError("請先呼叫 build_index()")
        
        # ── 第一階段：向量搜尋 ──
        query_embedding = self.bi_encoder.encode(
            [query], normalize_embeddings=True
        )
        
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            self.first_stage_k
        )
        
        # 取出候選文件
        candidate_docs = [self.documents[i] for i in indices[0]]
        candidate_scores = distances[0].tolist()
        
        print(f"\n第一階段召回 {len(candidate_docs)} 份候選文件")
        print(f"  Bi-Encoder 最高分: {max(candidate_scores):.4f}")
        print(f"  Bi-Encoder 最低分: {min(candidate_scores):.4f}")
        
        # ── 第二階段：Cross-Encoder 精排 ──
        # 構建 (query, document) pair 列表
        pairs = [(query, doc) for doc in candidate_docs]
        
        rerank_scores = self.cross_encoder.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False
        )
        
        # 合併原始排名資訊並重新排序
        results = [
            (candidate_docs[i], float(rerank_scores[i]), i)
            for i in range(len(candidate_docs))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 只返回前 n 份
        top_results = results[:self.final_top_n]
        
        print(f"\n第二階段 Rerank 後 Top-{self.final_top_n}：")
        for i, (doc, score, orig_rank) in enumerate(top_results):
            print(f"  [{i+1}] Rerank分數: {score:.4f} | 原始排名: #{orig_rank+1}")
            print(f"       {doc[:60]}...")
        
        return top_results


# ── 方案二：Cohere Rerank API ──

def cohere_rerank_example(query: str, documents: List[str], top_n: int = 5):
    """
    使用 Cohere Rerank API 進行精排。
    適合快速原型或不想自行部署模型的場景。
    """
    import cohere
    
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    
    response = co.rerank(
        model="rerank-multilingual-v3.0",  # 支援繁體中文
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True
    )
    
    print(f"\nCohere Rerank 結果：")
    for result in response.results:
        print(f"  [相關性分數: {result.relevance_score:.4f}]")
        print(f"  {result.document.text[:80]}...")
    
    return response.results


# ── NDCG 評估工具 ──

def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    計算 NDCG@k，評估排序品質。
    relevance_scores: 每個位置的相關性分數（0=不相關, 1=相關, 2=高度相關）
    """
    def dcg(scores, k):
        scores = scores[:k]
        return sum(
            (2**rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(scores)
        )
    
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True), k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── 效果評估示範 ──

def evaluate_reranker_effect():
    """
    模擬對比：加入 Reranker 前後的 NDCG@5 差異。
    """
    # 假設：10 份候選文件的真實相關性標籤 (0=不相關, 1=相關, 2=高度相關)
    true_relevance = [2, 0, 1, 0, 0, 2, 1, 0, 0, 1]
    
    # 向量搜尋返回的排序（按向量相似度）
    vector_search_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    vector_relevance = [true_relevance[i] for i in vector_search_order]
    
    # Reranker 重排後的順序（假設 Reranker 把最相關的提前）
    reranker_order = [0, 5, 2, 6, 9, 1, 3, 4, 7, 8]
    reranker_relevance = [true_relevance[i] for i in reranker_order]
    
    for k in [3, 5, 10]:
        ndcg_before = ndcg_at_k(vector_relevance, k)
        ndcg_after = ndcg_at_k(reranker_relevance, k)
        improvement = (ndcg_after - ndcg_before) / ndcg_before * 100
        print(f"NDCG@{k}: 無Reranker={ndcg_before:.4f} | 有Reranker={ndcg_after:.4f} | 提升={improvement:.1f}%")


# ── 使用範例 ──

if __name__ == "__main__":
    # 準備測試資料
    corpus = [
        "FAISS 是 Facebook AI 開發的高效向量相似度搜尋庫，支援十億級別向量的快速檢索。",
        "Python 的 list 是可變序列，支援 append、extend、insert 等操作。",
        "向量資料庫與傳統資料庫的核心區別：向量庫以 ANN 搜尋為主，傳統庫以精確匹配為主。",
        "RAG 系統的檢索品質受 chunk size、overlap 和 embedding 模型的影響最大。",
        "Transformer 的 attention 機制讓每個 token 都能關注序列中的其他 token。",
        "Chroma、Weaviate、Qdrant 是目前最流行的三款開源向量資料庫。",
        "如何選擇向量資料庫：考慮規模、延遲需求、是否需要 metadata 過濾、部署方式。",
        "機器學習中的過擬合問題可以用 Dropout、正規化等方法緩解。",
        "向量搜尋的近似最近鄰（ANN）演算法包括 HNSW、IVF、LSH 等。",
        "在生產環境部署 RAG 時，需要考慮索引更新頻率、向量庫 HA、回答延遲 SLA。",
    ]
    
    # 初始化兩階段檢索器
    retriever = TwoStageRetriever(
        bi_encoder_model="BAAI/bge-small-zh-v1.5",
        cross_encoder_model="BAAI/bge-reranker-base",
        first_stage_k=10,  # 全部候選
        final_top_n=3
    )
    
    retriever.build_index(corpus)
    results = retriever.retrieve("哪個向量資料庫比較好用？")
    
    print("\n" + "="*50)
    print("NDCG 評估示範：")
    evaluate_reranker_effect()
```

**安裝依賴**：
```bash
pip install sentence-transformers faiss-cpu numpy
# 若使用 Cohere API：
pip install cohere
```

### 實測效果資料

根據 BEIR（Benchmarking Information Retrieval）基準測試，加入 Cross-Encoder Reranker 後的典型提升幅度：

| 指標 | 僅向量搜尋 | 向量搜尋 + Reranker | 提升幅度 |
|------|----------|-------------------|---------|
| NDCG@10 | 0.43 | 0.56 | +30% |
| MRR@10 | 0.38 | 0.51 | +34% |
| Recall@5 | 0.61 | 0.61 | 持平（召回不變） |
| Precision@3 | 0.44 | 0.62 | +41% |

**關鍵觀察**：Reranker 不改變召回率（Recall），因為它只是對已有候選集重排，並沒有引入新文件。它改善的是精準率——讓最相關的文件排到最前面。

## 優缺點分析

### 優點

**1. 精準度提升顯著**：Cross-Encoder 能充分理解查詢與文件的細粒度語義關係，NDCG@5 通常提升 25-40%。

**2. 直接改善 LLM 回答品質**：LLM 看到的 context 品質更高，最終答案準確性明顯改善，幻覺率降低。

**3. 可插拔架構**：Reranker 作為獨立模組，不需要修改向量庫或嵌入模型，可以直接插入現有 RAG 管道。

**4. 支援多語言**：Cohere Rerank-multilingual 和 BGE-Reranker 都支援中英文，可直接用於繁體中文場景。

### 缺點

**1. 增加端到端延遲**：對 50 份候選文件進行 Cross-Encoder 推理，GPU 上約需 50-200ms，CPU 上可能達到 1-2 秒。

**2. 計算成本線性增加**：候選文件數量 × 每次推理成本。若第一階段取 Top-100，Reranker 就需對 100 個 pair 各推理一次。

**3. 模型部署複雜度**：本地部署需要 GPU 資源，使用 Cohere API 有隱私和成本考量。

**4. 對短文本效果有限**：文件 chunk 太短時，Cross-Encoder 的優勢無法充分展現（沒有足夠的語義內容可交互）。

## 適用場景

**高精準度問答（法律、醫療、金融）**：這些領域對「找對文件」的要求極高，一份不相關的文件可能導致嚴重錯誤的回答，Reranker 是必備模組。

**使用者問題複雜度高的場景**：多條件複合問題（「這個藥和那個藥可以一起吃嗎，孕婦適用嗎」），向量搜尋的相似度不足以捕捉複雜的相關性條件，Cross-Encoder 能精確建模。

**知識庫中存在大量相似文件**：多份文件語義相近但細節不同時（如同一產品的不同版本文件），Reranker 能區分「最相關的那份」。

**結果品質比延遲更重要的場景**：非同步報告生成、離線摘要、API 呼叫容忍 1-2 秒延遲的場合，都可以大方使用 Reranker。

## 與其他方法的比較

| 方法 | 改善目標 | 技術機制 | 對 Latency 的影響 |
|------|---------|---------|-----------------|
| **Reranker** | 精準率 | Cross-Encoder 交互注意力 | +50-200ms |
| **Multi-Query** | 召回率 | 多角度查詢擴展 | +LLM 呼叫延遲 |
| **Document Augmentation** | 召回率 | 多視圖索引 | 無（預處理） |
| **RSE（文章8）** | 精準率+節省 Token | 精確片段提取 | +小幅 LLM 呼叫 |

**Reranker vs. Multi-Query**：互補關係。Multi-Query 提升「找得到的量」（召回），Reranker 提升「排在前面的質」（精準）。最強配置是兩者組合：Multi-Query 擴大候選集，Reranker 從大候選集中精選最佳結果。

## 小結

Reranker 是 RAG 系統中效果最立竿見影的精準度優化手段。兩階段架構（Bi-Encoder + Cross-Encoder）解決了速度與精準度的根本矛盾，讓系統既能快速從百萬文件中召回候選，又能精準找到最相關的幾份。

實作建議：

1. **先用 Cohere Rerank API 驗證效果**：無需部署，10 分鐘內可以看到效果差異，確認值得投入後再考慮自建。
2. **第一階段多取一點**：設 Top-50 或 Top-100，給 Reranker 足夠的候選空間；不要因為怕慢就縮到 Top-10，那樣召回率就固定住了。
3. **最終只送 Top-3 到 Top-5 給 LLM**：Reranker 精排後不需要太多文件，Top-3 通常足夠，過多 context 反而讓 LLM 分心。
4. **對延遲敏感的場景用 Flashrank 或 MiniLM**：這些輕量模型在 CPU 上也能 50ms 內完成 20 份文件的精排，效果稍遜但部署門檻極低。
