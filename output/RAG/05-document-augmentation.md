---
title: "Document Augmentation：用多視圖策略讓文件更容易被搜尋到"
description: "原始文件對向量搜尋不友善——太長、術語不統一、視角單一。Document Augmentation 透過自動生成摘要、FAQ、假設問題，大幅提升 RAG 的語義覆蓋率。"
date: "2025-01-15"
tags: ["RAG", "Document Augmentation", "HyQ", "向量搜尋", "LLM", "知識庫", "語義搜尋"]
---

# Document Augmentation：用多視圖策略讓文件更容易被搜尋到

一篇技術規格書寫得很完整，但使用者問「這個 API 有哪些限制？」，卻找不到任何結果——不是文件沒寫，而是文件用的是「配額上限為每日 1000 次」這樣的表述，和查詢的語義對不上。Document Augmentation 就是解決這個問題的方法：在索引階段就替每份文件生成多種「視角」，讓原本難以被搜尋到的內容浮出水面。

## 核心概念

**Document Augmentation（文件增強）** 的核心假設是：原始文件的撰寫視角（作者視角、說明性語言）和使用者查詢的視角（問題視角、口語化語言）天生不匹配。

解法是在索引階段，用 LLM 替每份文件自動生成多種補充性表示（Supplementary Views），然後將這些視圖一起存入向量庫。檢索時，即使使用者的查詢和原文措辭相差甚遠，也能透過「補充視圖」命中正確文件。

三種最常用的增強形式：

| 增強類型 | 輸出形式 | 解決問題 |
|---------|---------|---------|
| **摘要（Summary）** | 100-200 字的精簡描述 | 文件過長、主旨分散 |
| **FAQ 問答列表** | 3-5 個問答對 | 文件以陳述式寫作，不利問句查詢 |
| **假設使用者問題（HyQ）** | 5-10 個使用者可能問的問題 | 查詢語義與文件語義不對齊 |

## 運作原理

Document Augmentation 在 RAG 管道的**索引階段**介入，不影響線上推理流程：

```
原始文件集
    │
    ▼
┌───────────────────────────────┐
│    Document Augmentation     │
│                               │
│  文件 → LLM 生成增強內容      │
│         ├── 摘要              │
│         ├── FAQ 問答          │
│         └── 假設問題 (HyQ)   │
└───────────────────────────────┘
    │
    ▼
向量化（原文 + 增強內容分別 Embedding）
    │
    ▼
向量資料庫
    │
    ├── 索引：增強內容的 Embedding
    └── 儲存：對應原文 chunk（作為 payload）

─────── 線上查詢流程 ───────

使用者查詢
    │
    ▼
查詢 Embedding → 向量搜尋（比對增強內容）
    │
    ▼
返回原文 chunk（非增強內容）
    │
    ▼
LLM 生成最終答案
```

**關鍵設計決策**：索引增強內容的 Embedding，但回傳原文 chunk 給 LLM。這樣既提升了召回率，又確保 LLM 看到的是完整、精確的原始資訊。

### 索引策略對比

**策略一：原文 + 增強內容全部索引**
- 優點：召回率最高
- 缺點：儲存空間增加 3-5 倍，可能引入重複結果

**策略二：只索引增強內容，儲存原文**
- 優點：搜尋空間更純淨，結果不重複
- 缺點：若增強品質差，原文直接被遮蔽

**策略三：分層索引（推薦）**
- 建立兩個索引：原文索引 + HyQ 索引
- 查詢時各取 Top-k，合併後去重
- 實際效果最佳，靈活性高

## Python 實作範例

以下範例示範如何用 GPT-4o-mini 批量生成文件的假設問題（HyQ），並存入 ChromaDB 向量庫。

```python
import os
import json
from typing import List, Dict
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 初始化 ChromaDB（本地模式）
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)

# 建立兩個 collection：原文索引 + HyQ 索引
original_collection = chroma_client.get_or_create_collection(
    name="original_docs",
    embedding_function=embedding_fn
)
hyq_collection = chroma_client.get_or_create_collection(
    name="hyq_docs",
    embedding_function=embedding_fn
)


def generate_hypothetical_questions(doc_text: str, n_questions: int = 5) -> List[str]:
    """
    用 GPT-4o-mini 為文件生成假設使用者問題（HyQ）。
    返回問題列表，每個問題都代表使用者可能用來找這份文件的查詢方式。
    """
    prompt = f"""你是一位技術文件分析師。請閱讀以下文件片段，
然後生成 {n_questions} 個使用者「可能會問」的問題。
這些問題應該：
- 用口語化、自然語言撰寫
- 涵蓋文件的不同重點
- 反映真實使用者的查詢習慣

文件內容：
{doc_text}

請以 JSON 陣列格式輸出，例如：
["問題1", "問題2", "問題3"]

只輸出 JSON，不要其他說明。"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    # 解析回傳的 JSON
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        # 相容不同的 JSON 結構
        if isinstance(data, list):
            return data
        # 有時 GPT 會包在 key 裡
        for v in data.values():
            if isinstance(v, list):
                return v
    except json.JSONDecodeError:
        pass
    return []


def generate_summary(doc_text: str) -> str:
    """為文件生成 100 字以內的摘要，強調核心功能與用途。"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"請用 100 字以內總結以下文件的核心內容，重點說明功能與用途：\n\n{doc_text}"
        }],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def augment_and_index(documents: List[Dict[str, str]]):
    """
    批量處理文件：生成增強內容並存入向量庫。
    
    documents 格式：[{"id": "doc_001", "text": "...", "metadata": {...}}, ...]
    """
    for doc in documents:
        doc_id = doc["id"]
        doc_text = doc["text"]
        metadata = doc.get("metadata", {})
        
        print(f"處理文件 {doc_id}...")
        
        # 1. 存入原文索引
        original_collection.add(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{**metadata, "type": "original"}]
        )
        
        # 2. 生成假設問題並索引
        questions = generate_hypothetical_questions(doc_text, n_questions=5)
        for i, question in enumerate(questions):
            hyq_id = f"{doc_id}_hyq_{i}"
            hyq_collection.add(
                ids=[hyq_id],
                documents=[question],
                # 關鍵：metadata 中保存原文，讓後續可以取回
                metadatas=[{
                    **metadata,
                    "type": "hyq",
                    "source_doc_id": doc_id,
                    "original_text": doc_text[:500]  # 截取前 500 字作為預覽
                }]
            )
        
        print(f"  ✓ 原文已索引，生成了 {len(questions)} 個假設問題")


def augmented_retrieval(query: str, top_k: int = 3) -> List[Dict]:
    """
    雙索引檢索：同時查詢原文索引和 HyQ 索引，合併結果後去重。
    """
    # 查詢原文索引
    original_results = original_collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # 查詢 HyQ 索引
    hyq_results = hyq_collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # 合併並去重（以 source_doc_id 為基準）
    seen_doc_ids = set()
    merged = []
    
    # 先加入原文結果
    for i, doc_id in enumerate(original_results["ids"][0]):
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            merged.append({
                "doc_id": doc_id,
                "text": original_results["documents"][0][i],
                "source": "original",
                "distance": original_results["distances"][0][i]
            })
    
    # 再加入 HyQ 命中的原文
    for i, hyq_id in enumerate(hyq_results["ids"][0]):
        meta = hyq_results["metadatas"][0][i]
        source_doc_id = meta.get("source_doc_id", hyq_id)
        
        if source_doc_id not in seen_doc_ids:
            seen_doc_ids.add(source_doc_id)
            merged.append({
                "doc_id": source_doc_id,
                "text": meta.get("original_text", ""),
                "source": "hyq",
                "matched_question": hyq_results["documents"][0][i],
                "distance": hyq_results["distances"][0][i]
            })
    
    return merged


# ── 使用範例 ──
if __name__ == "__main__":
    # 準備測試文件
    sample_docs = [
        {
            "id": "api_rate_limit",
            "text": """API 配額管理
本服務對每個 API Token 實施請求頻率限制，以確保服務穩定性。
免費方案：每日最多 1,000 次請求，每分鐘不超過 60 次。
付費方案：每日最多 100,000 次請求，每分鐘不超過 600 次。
超出限制後，伺服器將回傳 429 Too Many Requests 狀態碼。
建議實作指數退避（Exponential Backoff）重試機制。""",
            "metadata": {"category": "API 文件", "version": "v2"}
        },
        {
            "id": "auth_guide",
            "text": """身份驗證指南
所有 API 請求必須在 Header 中包含有效的認證資訊。
支援兩種驗證方式：
1. Bearer Token：在 Authorization header 加入 Bearer <your_token>
2. API Key：在 X-API-Key header 傳入金鑰
Token 有效期為 24 小時，過期後需重新取得。
建議將 Token 儲存在環境變數中，避免硬編碼在程式碼內。""",
            "metadata": {"category": "安全", "version": "v2"}
        }
    ]
    
    # 批量增強並索引
    augment_and_index(sample_docs)
    
    # 測試查詢：這個問法和原文措辭差異明顯
    test_queries = [
        "呼叫次數有上限嗎？",         # 原文寫「配額」和「頻率限制」
        "怎麼在 Header 帶上 token？",  # 原文寫「Authorization」
    ]
    
    for query in test_queries:
        print(f"\n查詢：{query}")
        results = augmented_retrieval(query, top_k=2)
        for r in results:
            print(f"  [{r['source']}] {r['doc_id']} (距離: {r['distance']:.4f})")
            if r['source'] == 'hyq':
                print(f"  匹配問題：{r.get('matched_question', '')}")
```

**執行前請安裝依賴**：
```bash
pip install openai chromadb
```

## 優缺點分析

### 優點

**1. 顯著提升語義覆蓋率**：同一份文件從多個語義角度被索引，有效縮小查詢和文件之間的「語義鴻溝」。

**2. 對使用者透明**：增強過程完全在後端，使用者的查詢體驗沒有任何改變。

**3. 模組化、可獨立優化**：增強品質（Prompt 設計）和索引策略可以分開調整，不影響系統其他部分。

**4. 適用現有架構**：不需要改變 LLM 推理流程，只需在索引管道加入一個步驟。

### 缺點

**1. 預處理成本高**：每份文件需要額外的 LLM 呼叫（生成增強內容），對大型知識庫成本顯著。100 萬份文件，每份 5 個問題 = 500 萬次 LLM 呼叫。

**2. 增強品質依賴 Prompt**：如果 Prompt 設計不好，生成的假設問題過於相似或偏離主題，效果反而不如原文搜尋。

**3. 索引維護複雜度增加**：文件更新時，所有對應的增強內容也需要重新生成，版本管理更複雜。

**4. 增加向量庫儲存需求**：若同時索引原文和增強內容，儲存量增加 3-5 倍。

## 適用場景

**產品文件與技術 FAQ**：開發者查詢時習慣用問題形式，而文件通常以功能描述形式撰寫。HyQ 能有效橋接這個差距。

**客服知識庫**：客服文件多以政策、流程描述撰寫，但客戶的提問語言口語化且多元。Document Augmentation 可大幅降低「找不到答案」的情況。

**法律/合規文件**：法規原文晦澀難懂，但業務端需要快速查詢「某種情況下合不合規」。摘要和 FAQ 增強讓非法律專業人員也能準確找到相關條文。

**多語言知識庫**：生成的增強內容可以跨語言，讓英文文件也能被中文查詢找到。

**文件更新頻率低的場景**：若文件不常改動，預處理成本只需付出一次，長期 ROI 很高。

## 與其他方法的比較

| 方法 | 介入階段 | 核心機制 | 效果 | 成本 |
|------|---------|---------|------|------|
| **Document Augmentation** | 索引期 | 多視圖文件表示 | 提升召回率 | 高（一次性預處理） |
| **Query Transformation** | 查詢期 | 多視角查詢改寫 | 提升召回率 | 中（每次查詢） |
| **Reranker** | 查詢後 | 精排Top-k結果 | 提升精準率 | 中（每次查詢） |
| **Chunking 優化** | 索引期 | 改善分塊策略 | 提升結構完整性 | 低（一次性） |

**Document Augmentation vs. Query Transformation**：這兩者是互補關係，都是解決「語義不匹配」，但前者在索引端處理，後者在查詢端處理。組合使用效果最佳，但成本也最高。

**Document Augmentation vs. Reranker**：目標不同。Augmentation 提升的是「有沒有找到」（召回率），Reranker 改善的是「找到的排在前面嗎」（精準率）。實際系統通常兩者都用。

## 小結

Document Augmentation 的核心價值是**在索引時付出一次較高的成本，換取長期的查詢品質提升**。對於查詢模式多樣、文件語言和使用者查詢語言差距大的場景，這個策略的 ROI 非常高。

實作建議：

1. **先從 HyQ 開始**：假設問題的增強效果通常最顯著，成本也相對合理。
2. **批量非同步處理**：使用 async 批量呼叫 LLM，配合速率限制控制，可大幅降低索引時間。
3. **評估 Prompt 品質**：抽取 20-30 份文件，人工檢查生成的問題是否真實反映潛在查詢，再大規模執行。
4. **策略三（分層索引）適合大多數場景**：建立獨立的 HyQ 索引，保留原文索引，合併查詢結果後去重，是成本與效果最平衡的方案。
