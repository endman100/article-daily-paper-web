---
title: "Semantic Chunking：讓 RAG 不再切斷語義"
description: "固定大小切分會在第 512 個 token 把一個完整論述劈成兩半。Semantic Chunking 用 embedding 的餘弦距離找語義邊界，讓每個 chunk 保有完整意思。本文說明原理、三種閾值模式，並附完整 Python 範例。"
date: 2025-01-16
tags: ["RAG", "Semantic Chunking", "LangChain", "Embedding", "NLP", "文字處理"]
---

# Semantic Chunking：讓 RAG 不再切斷語義

考慮這個具體場景：一份技術報告的第三段同時描述「資料庫效能問題的成因」和「對應的索引優化方案」。用 Fixed-size chunking（`chunk_size=512`）處理後，前半段（問題描述）進了 chunk #7，後半段（解決方案）進了 chunk #8。使用者問「如何解決這個效能問題？」，向量搜尋只找回 chunk #7——只有問題，沒有答案。

這不是偶發的邊界運氣問題，而是固定大小切分的結構性缺陷。Semantic Chunking 的解法是：**不按字數切，按語義變化切**。

---

## 核心概念

Fixed-size chunking 把文件視為一條均勻的 token 串，每隔 N 個 token 就切一刀，視需要加上若干 token 的重疊（overlap）。這個做法快速、無需任何模型，但它對語言結構一無所知。

Semantic Chunking 的核心假設是：**同一個 chunk 內的句子應該在語義上互相關聯；當相鄰句子群組的語義差異突然升高，就是切分點**。

衡量語義差異的工具是 **cosine distance**（餘弦距離）：

$$\text{cosine\_distance}(A, B) = 1 - \frac{A \cdot B}{\|A\| \|B\|}$$

距離越接近 0，兩個 embedding 越相似；越接近 1，差異越大。

---

## 運作原理

演算法分四步驟，以下用 ASCII 示意：

```
原始文件
   │
   ▼
┌─────────────────────────────────────┐
│  步驟 1：句子切分                    │
│  "A. B. C. D. E. F. G."             │
│   → [A] [B] [C] [D] [E] [F] [G]    │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  步驟 2：滑動視窗組群 + Embedding    │
│  視窗大小 = 1（每側各1句）           │
│  group_0 = [A, B, C]  → emb_0      │
│  group_1 = [B, C, D]  → emb_1      │
│  group_2 = [C, D, E]  → emb_2      │
│  group_3 = [D, E, F]  → emb_3      │
│  group_4 = [E, F, G]  → emb_4      │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  步驟 3：計算相鄰群組的 cosine dist  │
│  dist(0→1) = 0.12  ─ 低，同主題     │
│  dist(1→2) = 0.15  ─ 低，同主題     │
│  dist(2→3) = 0.71  ◄─ 高！主題轉換  │
│  dist(3→4) = 0.09  ─ 低，同主題     │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  步驟 4：閾值判斷，決定切分點        │
│  閾值 = 0.5（由選定模式計算）        │
│  dist(2→3) = 0.71 > 0.5 → 切！      │
│                                     │
│  Chunk 1: [A, B, C, D]              │
│  Chunk 2: [E, F, G]                 │
└─────────────────────────────────────┘
```

關鍵在步驟 3 和步驟 4：距離陣列裡的「突波」就是語義邊界。如何定義「夠高才算突波」，由以下三種閾值模式決定。

### 三種閾值計算模式

**1. Percentile（百分位數）**
取所有相鄰距離值的第 *p* 百分位（預設 95）作為閾值。適合距離分布不均勻的文件，能自動適應文件整體的語義密度。

```
distances = [0.12, 0.15, 0.71, 0.09, 0.63, 0.08]
threshold = np.percentile(distances, 95)  # ≈ 0.69
```

**2. Standard Deviation（標準差）**
閾值 = 平均值 + N × 標準差（預設 N=3）。當距離大致呈常態分布時效果穩定；對離群的語義跳躍非常敏感。

```
threshold = mean(distances) + 3 * std(distances)
```

**3. Interquartile Range（四分位距）**
閾值 = Q3 + 1.5 × IQR，IQR = Q3 − Q1。這是統計學中的經典離群值偵測方法，對非對稱分布有較強的魯棒性。

```
Q1, Q3 = np.percentile(distances, [25, 75])
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR
```

---

## Python 實作範例

### 方法一：使用 LangChain SemanticChunker（推薦用於生產）

```python
# 安裝套件：pip install langchain langchain-openai langchain-experimental

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 示範用的長文件（混合兩個不同主題）
sample_text = """
資料庫查詢效能下降的主因通常是缺少適當的索引。
當資料表的資料量超過百萬筆，全表掃描（Full Table Scan）的成本會呈線性成長。
建立複合索引（Composite Index）時，欄位的順序至關重要，應將選擇性高的欄位放在前面。
EXPLAIN 指令可以幫助分析查詢計畫，確認索引是否被正確使用。

量子運算利用量子位元（qubit）的疊加態和糾纏態來處理資訊。
不同於傳統位元只能是 0 或 1，量子位元可以同時處於多種狀態的疊加。
Shor 演算法展示了量子電腦可以在多項式時間內分解大整數，這對現有加密體系構成威脅。
目前主要的量子運算平台包括 IBM Quantum、Google Sycamore 和 IonQ。
"""

embeddings = OpenAIEmbeddings()

# ── 模式 1：Percentile（百分位數）──────────────────────────────────────
chunker_percentile = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",   # 使用百分位數模式
    breakpoint_threshold_amount=95,           # 取第 95 百分位作為閾值
)

chunks_p = chunker_percentile.create_documents([sample_text])
print("=== Percentile 模式 ===")
for i, chunk in enumerate(chunks_p):
    print(f"Chunk {i+1} ({len(chunk.page_content)} chars):\n{chunk.page_content[:100]}...\n")

# ── 模式 2：Standard Deviation（標準差）────────────────────────────────
chunker_std = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",  # 使用標準差模式
    breakpoint_threshold_amount=3,                   # 平均值 + 3 個標準差
)

chunks_s = chunker_std.create_documents([sample_text])
print("=== Standard Deviation 模式 ===")
for i, chunk in enumerate(chunks_s):
    print(f"Chunk {i+1} ({len(chunk.page_content)} chars):\n{chunk.page_content[:100]}...\n")

# ── 模式 3：Interquartile Range（四分位距）─────────────────────────────
chunker_iqr = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="interquartile",  # 使用四分位距模式
    breakpoint_threshold_amount=1.5,            # Q3 + 1.5 × IQR
)

chunks_i = chunker_iqr.create_documents([sample_text])
print("=== Interquartile 模式 ===")
for i, chunk in enumerate(chunks_i):
    print(f"Chunk {i+1} ({len(chunk.page_content)} chars):\n{chunk.page_content[:100]}...\n")
```

### 方法二：手動實作（理解底層邏輯）

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """呼叫 OpenAI API 取得文字的 embedding 向量"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def semantic_chunk(text: str, threshold_percentile: float = 95) -> list[str]:
    """
    手動實作 Semantic Chunking，使用 Percentile 閾值模式。

    Args:
        text: 要切分的原始文件
        threshold_percentile: 切分閾值的百分位數（預設 95）

    Returns:
        切分後的 chunk 列表
    """
    # 步驟 1：按句號切成句子列表（實務上建議用 NLTK 或 spaCy）
    sentences = [s.strip() for s in text.replace("。", "。\n").split("\n") if s.strip()]
    print(f"共 {len(sentences)} 個句子")

    # 步驟 2：為每個句子計算 embedding
    print("計算句子 embeddings...")
    sentence_embeddings = [get_embedding(s) for s in sentences]

    # 步驟 3：建立滑動視窗組群的 embedding（視窗大小 = 前後各1句）
    group_embeddings = []
    window = 1  # 每側視窗大小
    for i in range(len(sentences)):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        # 將視窗內的 embedding 取平均，代表該局部語義
        group_emb = np.mean(sentence_embeddings[start:end], axis=0)
        group_embeddings.append(group_emb)

    # 步驟 4：計算相鄰群組的 cosine distance
    distances = []
    for i in range(len(group_embeddings) - 1):
        emb_a = np.array(group_embeddings[i]).reshape(1, -1)
        emb_b = np.array(group_embeddings[i + 1]).reshape(1, -1)
        similarity = cosine_similarity(emb_a, emb_b)[0][0]
        distance = 1 - similarity  # cosine distance = 1 - cosine similarity
        distances.append(distance)

    print(f"距離分布：min={min(distances):.3f}, max={max(distances):.3f}, "
          f"mean={np.mean(distances):.3f}")

    # 步驟 5：用百分位數決定切分閾值
    threshold = np.percentile(distances, threshold_percentile)
    print(f"第 {threshold_percentile} 百分位閾值：{threshold:.3f}")

    # 步驟 6：在超過閾值的位置切分
    split_points = [i + 1 for i, d in enumerate(distances) if d > threshold]
    print(f"切分點位置（句子索引）：{split_points}")

    # 步驟 7：依切分點組合成 chunk
    chunks = []
    prev = 0
    for point in split_points:
        chunk_sentences = sentences[prev:point]
        chunks.append(" ".join(chunk_sentences))
        prev = point
    chunks.append(" ".join(sentences[prev:]))  # 最後一個 chunk

    return chunks


# ── 執行範例 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    text = """
    機器學習模型的訓練過程需要大量的標注資料。
    資料品質直接影響模型的泛化能力，噪聲資料會導致模型過擬合。
    交叉驗證是評估模型效能的標準方法，常用 K-Fold 交叉驗證。
    超參數調整可以透過 Grid Search 或 Bayesian Optimization 完成。

    氣候變遷導致全球平均氣溫持續上升，極端天氣事件頻率增加。
    碳排放是主要驅動因素，工業、交通和農業是三大排放來源。
    再生能源的成本在過去十年間大幅下降，太陽能發電成本降低了 90%。
    各國政府正在推進碳中和目標，歐盟預計在 2050 年實現淨零排放。
    """

    result_chunks = semantic_chunk(text.strip(), threshold_percentile=90)

    print(f"\n共切出 {len(result_chunks)} 個 chunks：")
    for i, chunk in enumerate(result_chunks):
        print(f"\n── Chunk {i+1} ──")
        print(chunk)
```

---

## 優缺點分析

| 面向 | Semantic Chunking | Fixed-size Chunking |
|------|-------------------|---------------------|
| **語義完整性** | ✅ 保持主題內聚性，切分點在語義邊界 | ❌ 按字數截斷，可能切斷論述 |
| **計算成本** | ❌ 高：每個句子都需要呼叫 embedding API | ✅ 低：純字串操作，無需模型 |
| **Chunk 大小一致性** | ❌ 大小不均（短主題 vs 長主題）| ✅ 大小可控，便於管理向量索引 |
| **實作複雜度** | ❌ 較高，需要選擇和調校閾值模式 | ✅ 低，只需指定 `chunk_size` |
| **跨句子關聯保留** | ✅ 優秀 | ⚠️ 依賴 overlap，仍有遺漏風險 |
| **文件類型適應性** | ✅ 自動適應不同主題密度的文件 | ❌ 對所有文件一視同仁 |
| **Retrieval 準確率** | ✅ 通常更高（主題純粹的 chunk）| ⚠️ 受切分運氣影響 |
| **適合文件長度** | ✅ 長文件表現佳 | ✅ 短、長皆可 |
| **API 費用** | ❌ 建索引時費用顯著高於 Fixed-size | ✅ 建索引不需要額外 embedding 費用 |

**閾值模式的適用情境比較**：

| 閾值模式 | 適合情境 | 產生 chunk 數量 |
|---------|---------|----------------|
| Percentile (95) | 通用場景，文件主題數未知 | 中等（約 5% 的位置被切）|
| Standard Deviation (3σ) | 學術論文、段落明確的技術文件 | 較少（只切最明顯的跳躍）|
| Interquartile | 新聞、博客等主題分散的文章 | 較多（對適中差異也切分）|

---

## 適用場景

**高度推薦使用 Semantic Chunking 的情境**：

1. **學術論文**：一篇論文涵蓋背景、方法、實驗、討論四大部分，每部分的語義距離明顯，Semantic Chunking 通常能自然地在章節邊界切分，不需要依賴文件結構標記（標題、換行）。

2. **法律文件**：合約中的各條款討論不同義務，若被 Fixed-size 切斷，RAG 系統可能把「甲方義務」和「乙方義務」混入同一個 chunk，誤導 LLM 的答案。

3. **技術文檔**：API 文件中，一個函數的「參數說明」和「回傳值說明」緊密相關，應在同一 chunk；不同函數之間語義差異大，應切分。

4. **長篇報告（> 5000 字）**：文件越長，Fixed-size 遭遇語義邊界的機率越高，Semantic Chunking 的相對優勢越明顯。

5. **多主題混合文件**：例如公司年報同時包含財務數據、業務描述、風險因素，語義邊界清晰。

**不推薦的情境**：
- 文件本身很短（< 500 字），句子數不足以讓閾值計算有統計意義。
- 需要嚴格控制 chunk token 數以符合 LLM context window 限制時，應使用 Fixed-size 或在 Semantic Chunking 後加上大小過濾。
- 預算有限且文件量極大（數百萬份短文件）的批次建索引任務。

---

## 與其他方法的比較

用同一段文字說明兩種方法的實際差異：

**原始文字（共約 280 字）**：

> 傳統的關聯式資料庫在結構化查詢上表現出色，但面對非結構化資料（如文字、圖片）時能力有限。NoSQL 資料庫應運而生，提供更靈活的資料模型。MongoDB 使用文件模型，Redis 提供鍵值對儲存，Neo4j 則專注於圖資料庫。這三種系統分別針對不同的使用場景進行了優化。
>
> 向量資料庫是近年興起的另一類資料庫，專為儲存和搜尋高維向量而設計。Pinecone、Weaviate、Qdrant 是目前主流的向量資料庫產品。它們的核心操作是近似最近鄰搜尋（ANN），透過 HNSW 或 IVF 等索引結構達到毫秒級查詢速度。向量資料庫是 RAG 系統的關鍵基礎設施。

**Fixed-size chunking（chunk_size=100 tokens, overlap=20）的結果**：

```
Chunk 1: 傳統的關聯式資料庫在結構化查詢上表現出色，但面對非結構化資料
         （如文字、圖片）時能力有限。NoSQL 資料庫應運而生，提供更靈活的
         資料模型。MongoDB 使用文件模型，Redis 提供鍵值對儲存，Neo4j 則

Chunk 2: Neo4j 則專注於圖資料庫。這三種系統分別針對不同的使用場景進行了
         優化。向量資料庫是近年興起的另一類資料庫，專為儲存和搜尋高維向量

Chunk 3: 高維向量而設計。Pinecone、Weaviate、Qdrant 是目前主流的向量資料庫
         產品。它們的核心操作是近似最近鄰搜尋（ANN），透過 HNSW 或 IVF 等
         索引結構達到毫秒級查詢速度。向量資料庫是 RAG 系統的關鍵基礎設施。
```

**問題**：Chunk 2 同時包含「NoSQL 資料庫的結語」和「向量資料庫的開頭」，兩個主題混在一起。使用者問「什麼是向量資料庫？」時，Chunk 2 的前半段對答案毫無幫助，卻佔用了 LLM 的 context window。

---

**Semantic Chunking 的結果**：

```
Chunk 1: 傳統的關聯式資料庫在結構化查詢上表現出色，但面對非結構化資料
         （如文字、圖片）時能力有限。NoSQL 資料庫應運而生，提供更靈活的
         資料模型。MongoDB 使用文件模型，Redis 提供鍵值對儲存，Neo4j 則
         專注於圖資料庫。這三種系統分別針對不同的使用場景進行了優化。

Chunk 2: 向量資料庫是近年興起的另一類資料庫，專為儲存和搜尋高維向量而設計。
         Pinecone、Weaviate、Qdrant 是目前主流的向量資料庫產品。它們的核心
         操作是近似最近鄰搜尋（ANN），透過 HNSW 或 IVF 等索引結構達到毫秒
         級查詢速度。向量資料庫是 RAG 系統的關鍵基礎設施。
```

**結果**：切分點落在兩段的語義邊界（「NoSQL 討論」→「向量資料庫討論」的轉換處），每個 chunk 主題單一，retrieval 精準度顯著提升。

---

## 小結

選擇 chunking 策略時，以下三個問題能幫助快速決策：

1. **文件有多長？主題有多少？** 超過 2000 字、涵蓋 3 個以上主題的文件，優先考慮 Semantic Chunking。
2. **建索引的預算是否充裕？** Semantic Chunking 的 embedding 費用約為 Fixed-size 的 3–5 倍（因為要對每個句子做 embedding）。若文件量超過百萬份，先用 Fixed-size 驗證 RAG pipeline 的可行性，再針對高價值文件類型升級。
3. **閾值模式怎麼選？** 從 Percentile（95）開始，觀察切出的 chunk 數量是否合理（太多 → 降低百分位，太少 → 提高百分位）。

具體行動步驟：
- **立即可做**：用 `langchain_experimental.text_splitter.SemanticChunker` 在現有 RAG pipeline 上做 A/B 測試，對比 retrieval 的 MRR 或 Hit Rate。
- **調校優先順序**：先確認 embedding 模型的品質（`text-embedding-3-large` 通常優於 `text-embedding-3-small`），閾值模式的影響通常小於 embedding 模型的影響。
- **邊界仍不理想時**：結合文件結構（Markdown 標題、HTML tag）做 hierarchical chunking，以結構信號補充語義信號的不足。
