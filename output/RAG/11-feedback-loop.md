---
title: "Feedback Loop RAG：透過使用者反饋持續改善檢索與回答品質"
description: "RAG 系統上線後的效果往往不盡理想，且缺乏自我改善的機制。Feedback Loop 將使用者的點讚、點踩、修正行為轉化為訓練信號，用於優化 Embedding 模型、文件相關性權重和查詢快取，讓系統在使用中持續進化。"
date: 2025-01-11
tags: ["RAG", "Feedback Loop", "RAGAS", "Fine-tuning", "Embedding", "持續學習", "評估指標"]
---

# Feedback Loop RAG：透過使用者反饋持續改善檢索與回答品質

RAG 系統的初版通常是用靜態的 embedding 模型和固定的 chunking 策略建立的，這些設定在部署後就不再更新。然而使用者每天都在發出查詢、得到回答、判斷是否有用——這些訊號是提升系統效果的最珍貴數據，卻大多被丟棄。Feedback Loop 的核心思想是：收集使用者反饋，建立一個數據飛輪，讓系統的每個組件隨時間持續改善。

## 核心概念

### 為什麼需要 Feedback Loop？

RAG 系統面臨的根本挑戰是：**評估標準是主觀的，且隨時間變化**。

- 初期部署時沒有標注數據，embedding 模型可能不適合你的領域
- 用戶的查詢模式隨業務發展而改變
- 新文件的加入改變了原有的相關性分佈
- 同樣的問題，不同用戶對「好答案」的期望不同

沒有 Feedback Loop 的 RAG 系統就像一個從不接受批評的員工——第一天的表現決定了之後的上限。

### 三個反饋維度

**維度一：檢索反饋（Retrieval Feedback）**
- 用戶看到引用來源後，主動標記「這個來源有幫助 / 無關」
- 系統記錄哪些 chunk 被引用後得到正面反應
- 用途：
  - 更新 chunk 的「有用性權重」，影響下次檢索排名
  - 作為 embedding fine-tuning 的訓練數據（正樣本）

**維度二：答案反饋（Answer Feedback）**
- 點讚 / 點踩：最簡單的二元反饋
- 評分（1-5 星）：更細緻的品質評估
- 文字修正：用戶提供正確答案（最高品質的訓練數據）
- 用途：
  - 評估不同 RAG 策略的效果
  - 識別系統的薄弱點（哪類問題錯誤率高）
  - 建立問答快取（高分的 Q&A 對直接快取複用）

**維度三：隱性反饋（Implicit Feedback）**
不需要用戶主動操作，從行為中推斷：
- **點擊引用**：用戶點擊了某個來源 ↔ 這個 chunk 可能有用
- **複製文字**：用戶複製了回答的某段 ↔ 那段內容品質高
- **繼續追問**：用戶追問「還有嗎？」↔ 上一個答案不夠完整
- **立即關閉**：用戶看完就關閉 ↔ 答案令人滿意（或完全不滿意，難以區分）

### 反饋驅動改善的四條路徑

```
收集反饋
   │
   ├─→ 1. Fine-tune Embedding 模型（改善檢索相關性）
   │
   ├─→ 2. 更新文件/Chunk 相關性權重（即時影響排名）
   │
   ├─→ 3. 建立 Query-Answer 快取（降低成本、加速回應）
   │
   └─→ 4. 持續評估指標追蹤（RAGAS、自動化測試）
```

## 運作原理

### 整體架構

```
使用者介面
    │
    ├── 查詢 ──────────────────────────────────────────→ RAG 管道 → 回答
    │                                                        ↑
    │                                              (使用最新模型和權重)
    │
    ├── 反饋（👍👎 + 引用標記）────→ 反饋收集服務
    │                                      │
    │                              ┌───────┴───────┐
    │                              ↓               ↓
    │                        反饋資料庫        評估儀表板
    │                              │               │
    │                    ┌─────────┼─────────┐     │
    │                    ↓         ↓         ↓     │
    │              Chunk 權重  Q&A 快取  訓練數據集  │
    │              更新服務    更新服務   累積服務    │
    │                    │         │         │     │
    │                    └─────────┴────┬────┘     │
    │                                   ↓          │
    └──────────────────────── Embedding Fine-tuning │
                                        │           │
                                        ↓           │
                               新版 Embedding 模型  │
                                        │           │
                                        └───────────┘
                                         週期性部署
```

### RAGAS 自動評估框架

RAGAS（Retrieval-Augmented Generation Assessment）提供無需人工標注的自動評估，使用 LLM 作為評委：

| 指標 | 定義 | 計算方式 |
|------|------|---------|
| Faithfulness | 回答是否忠於檢索內容（不幻覺） | LLM 判斷每個陳述是否在 context 中有依據 |
| Answer Relevancy | 回答是否回應了問題 | LLM 從答案反向生成問題，計算與原問題的相似度 |
| Context Recall | 相關內容是否被檢索到 | 將標準答案的每個句子與 context 對照 |
| Context Precision | 檢索到的內容是否都相關 | 計算 context 中真正被用到的比例 |

## Python 實作範例

### 建立反饋收集系統

```python
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# ── 資料結構 ──────────────────────────────────────────────────────────────
@dataclass
class QueryRecord:
    """一次查詢的完整記錄"""
    query_id: str           # 唯一 ID
    query: str              # 用戶查詢
    retrieved_chunks: List[Dict]  # 檢索到的 chunks（含 chunk_id, content, score）
    answer: str             # 系統回答
    timestamp: str          # 查詢時間
    session_id: str         # 會話 ID（用於追蹤連續對話）
    latency_ms: int         # 回應延遲（毫秒）

@dataclass
class FeedbackRecord:
    """用戶反饋記錄"""
    feedback_id: str
    query_id: str           # 對應的查詢 ID
    rating: Optional[int]   # 1-5 星，None 表示未評分
    thumbs: Optional[str]   # "up" / "down" / None
    helpful_chunk_ids: List[str]   # 用戶標記有用的 chunk ID
    unhelpful_chunk_ids: List[str] # 用戶標記無用的 chunk ID
    correction: Optional[str]      # 用戶提供的正確答案
    timestamp: str

# ── SQLite 反饋資料庫 ─────────────────────────────────────────────────────
class FeedbackDatabase:
    def __init__(self, db_path: str = "rag_feedback.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """建立所有必要的表格"""
        self.conn.executescript("""
            -- 查詢記錄表
            CREATE TABLE IF NOT EXISTS query_logs (
                query_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                retrieved_chunks TEXT,  -- JSON
                answer TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                latency_ms INTEGER
            );

            -- 用戶反饋表
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                rating INTEGER,           -- 1-5
                thumbs TEXT,              -- 'up' / 'down'
                helpful_chunk_ids TEXT,   -- JSON array
                unhelpful_chunk_ids TEXT, -- JSON array
                correction TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES query_logs(query_id)
            );

            -- Chunk 相關性權重表（即時更新）
            CREATE TABLE IF NOT EXISTS chunk_weights (
                chunk_id TEXT PRIMARY KEY,
                helpful_count INTEGER DEFAULT 0,
                unhelpful_count INTEGER DEFAULT 0,
                last_updated TEXT,
                weight_score REAL DEFAULT 1.0  -- 初始權重為 1.0
            );

            -- 查詢答案快取表
            CREATE TABLE IF NOT EXISTS qa_cache (
                cache_key TEXT PRIMARY KEY,  -- query 的 hash
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_chunk_ids TEXT,       -- JSON
                avg_rating REAL DEFAULT 0,
                use_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_used TEXT
            );

            -- 訓練數據集（用於 embedding fine-tuning）
            CREATE TABLE IF NOT EXISTS training_pairs (
                pair_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                positive_chunk_id TEXT,  -- 有用的 chunk
                negative_chunk_id TEXT,  -- 無用的 chunk
                source TEXT,             -- 'explicit' / 'implicit'
                created_at TEXT
            );
        """)
        self.conn.commit()

    def log_query(self, record: QueryRecord):
        """記錄查詢"""
        self.conn.execute("""
            INSERT OR REPLACE INTO query_logs
            (query_id, query, retrieved_chunks, answer, timestamp, session_id, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.query_id,
            record.query,
            json.dumps(record.retrieved_chunks, ensure_ascii=False),
            record.answer,
            record.timestamp,
            record.session_id,
            record.latency_ms
        ))
        self.conn.commit()

    def log_feedback(self, feedback: FeedbackRecord):
        """記錄用戶反饋並觸發相關更新"""
        self.conn.execute("""
            INSERT OR REPLACE INTO user_feedback
            (feedback_id, query_id, rating, thumbs, helpful_chunk_ids,
             unhelpful_chunk_ids, correction, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.feedback_id,
            feedback.query_id,
            feedback.rating,
            feedback.thumbs,
            json.dumps(feedback.helpful_chunk_ids),
            json.dumps(feedback.unhelpful_chunk_ids),
            feedback.correction,
            feedback.timestamp,
        ))

        # 更新 chunk 相關性權重
        self._update_chunk_weights(
            feedback.helpful_chunk_ids,
            feedback.unhelpful_chunk_ids
        )

        # 若評分高，加入快取
        if feedback.rating and feedback.rating >= 4:
            self._update_cache(feedback.query_id, feedback.rating)

        # 產生訓練數據對
        if feedback.helpful_chunk_ids and feedback.unhelpful_chunk_ids:
            self._generate_training_pairs(
                feedback.query_id,
                feedback.helpful_chunk_ids,
                feedback.unhelpful_chunk_ids
            )

        self.conn.commit()

    def _update_chunk_weights(
        self,
        helpful_ids: List[str],
        unhelpful_ids: List[str]
    ):
        """更新 chunk 的相關性權重"""
        now = datetime.now().isoformat()
        for chunk_id in helpful_ids:
            self.conn.execute("""
                INSERT INTO chunk_weights (chunk_id, helpful_count, last_updated, weight_score)
                VALUES (?, 1, ?, 1.1)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    helpful_count = helpful_count + 1,
                    last_updated = ?,
                    -- 每個正反饋提升 10% 權重，上限 2.0
                    weight_score = MIN(2.0, weight_score * 1.1)
            """, (chunk_id, now, now))

        for chunk_id in unhelpful_ids:
            self.conn.execute("""
                INSERT INTO chunk_weights (chunk_id, unhelpful_count, last_updated, weight_score)
                VALUES (?, 1, ?, 0.9)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    unhelpful_count = unhelpful_count + 1,
                    last_updated = ?,
                    -- 每個負反饋降低 10% 權重，下限 0.3
                    weight_score = MAX(0.3, weight_score * 0.9)
            """, (chunk_id, now, now))

    def _update_cache(self, query_id: str, rating: int):
        """將高分 Q&A 加入快取"""
        row = self.conn.execute(
            "SELECT query, answer, retrieved_chunks FROM query_logs WHERE query_id = ?",
            (query_id,)
        ).fetchone()

        if not row:
            return

        query, answer, chunks_json = row
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        chunks = json.loads(chunks_json)
        chunk_ids = [c.get("chunk_id", "") for c in chunks]
        now = datetime.now().isoformat()

        self.conn.execute("""
            INSERT INTO qa_cache
            (cache_key, query, answer, source_chunk_ids, avg_rating, use_count, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                avg_rating = (avg_rating * use_count + ?) / (use_count + 1),
                use_count = use_count + 1,
                last_used = ?
        """, (cache_key, query, answer, json.dumps(chunk_ids), rating, now, now, rating, now))

    def _generate_training_pairs(
        self,
        query_id: str,
        helpful_ids: List[str],
        unhelpful_ids: List[str]
    ):
        """生成用於 Embedding fine-tuning 的正負樣本對"""
        row = self.conn.execute(
            "SELECT query FROM query_logs WHERE query_id = ?",
            (query_id,)
        ).fetchone()
        if not row:
            return

        query = row[0]
        now = datetime.now().isoformat()

        # 所有正負組合
        for pos_id in helpful_ids:
            for neg_id in unhelpful_ids:
                pair_id = hashlib.md5(f"{query_id}{pos_id}{neg_id}".encode()).hexdigest()
                self.conn.execute("""
                    INSERT OR IGNORE INTO training_pairs
                    (pair_id, query, positive_chunk_id, negative_chunk_id, source, created_at)
                    VALUES (?, ?, ?, ?, 'explicit', ?)
                """, (pair_id, query, pos_id, neg_id, now))

    def get_chunk_weight(self, chunk_id: str) -> float:
        """取得 chunk 的當前相關性權重"""
        row = self.conn.execute(
            "SELECT weight_score FROM chunk_weights WHERE chunk_id = ?",
            (chunk_id,)
        ).fetchone()
        return row[0] if row else 1.0

    def get_cached_answer(self, query: str, similarity_threshold: float = 0.95) -> Optional[str]:
        """查詢快取（精確匹配）"""
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        row = self.conn.execute(
            "SELECT answer FROM qa_cache WHERE cache_key = ? AND avg_rating >= 4",
            (cache_key,)
        ).fetchone()
        return row[0] if row else None

    def get_training_stats(self) -> Dict:
        """統計訓練數據集的狀況"""
        total_feedback = self.conn.execute("SELECT COUNT(*) FROM user_feedback").fetchone()[0]
        total_pairs = self.conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        avg_rating = self.conn.execute(
            "SELECT AVG(rating) FROM user_feedback WHERE rating IS NOT NULL"
        ).fetchone()[0]
        cache_size = self.conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]

        return {
            "total_feedback": total_feedback,
            "training_pairs": total_pairs,
            "average_rating": round(avg_rating or 0, 2),
            "cache_size": cache_size,
        }

# ── 帶反饋的 RAG 服務 ─────────────────────────────────────────────────────
import uuid
import time

class FeedbackAwareRAG:
    """整合反饋機制的 RAG 服務"""

    def __init__(self, retriever, llm, db: FeedbackDatabase):
        self.retriever = retriever
        self.llm = llm
        self.db = db

    def query(self, question: str, session_id: str = "") -> Dict:
        """
        執行查詢，自動整合快取和 chunk 權重
        """
        # 1. 檢查快取
        cached = self.db.get_cached_answer(question)
        if cached:
            return {
                "answer": cached,
                "source": "cache",
                "query_id": None,
                "chunks": []
            }

        # 2. 檢索
        start = time.time()
        raw_chunks = self.retriever.get_relevant_documents(question)

        # 3. 應用 chunk 權重重排
        weighted_chunks = []
        for chunk in raw_chunks:
            chunk_id = chunk.metadata.get("chunk_id", "")
            weight = self.db.get_chunk_weight(chunk_id)
            weighted_chunks.append((chunk, weight))

        # 按權重重排（原始分數 × 學習到的權重）
        weighted_chunks.sort(key=lambda x: x[1], reverse=True)
        final_chunks = [c for c, _ in weighted_chunks[:5]]

        # 4. 生成回答
        context = "\n\n".join(c.page_content for c in final_chunks)
        answer = self.llm.predict(
            f"根據以下資訊回答問題：\n\n{context}\n\n問題：{question}"
        )

        latency = int((time.time() - start) * 1000)

        # 5. 記錄查詢
        query_id = str(uuid.uuid4())
        record = QueryRecord(
            query_id=query_id,
            query=question,
            retrieved_chunks=[
                {
                    "chunk_id": c.metadata.get("chunk_id", ""),
                    "content": c.page_content[:200],
                    "score": c.metadata.get("score", 0)
                }
                for c in final_chunks
            ],
            answer=answer,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            latency_ms=latency
        )
        self.db.log_query(record)

        return {
            "answer": answer,
            "source": "rag",
            "query_id": query_id,
            "chunks": [c.metadata.get("chunk_id", "") for c in final_chunks]
        }

    def submit_feedback(
        self,
        query_id: str,
        thumbs: str,
        rating: Optional[int] = None,
        helpful_chunk_ids: List[str] = None,
        unhelpful_chunk_ids: List[str] = None,
        correction: Optional[str] = None
    ):
        """提交用戶反饋"""
        feedback = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            query_id=query_id,
            rating=rating,
            thumbs=thumbs,
            helpful_chunk_ids=helpful_chunk_ids or [],
            unhelpful_chunk_ids=unhelpful_chunk_ids or [],
            correction=correction,
            timestamp=datetime.now().isoformat()
        )
        self.db.log_feedback(feedback)
        return {"status": "ok", "feedback_id": feedback.feedback_id}
```

### RAGAS 自動評估整合

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # 忠實度：回答是否基於 context
    answer_relevancy,      # 相關性：回答是否回應問題
    context_recall,        # 召回率：相關內容是否被檢索到
    context_precision,     # 精確度：context 是否都相關
)
from datasets import Dataset

def evaluate_rag_system(
    questions: List[str],
    ground_truths: List[str],
    rag_service: FeedbackAwareRAG
) -> Dict:
    """
    使用 RAGAS 評估 RAG 系統效果
    
    Args:
        questions: 測試問題列表
        ground_truths: 標準答案列表
        rag_service: 待評估的 RAG 服務
    
    Returns:
        各指標的平均分數
    """
    results = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for question, ground_truth in zip(questions, ground_truths):
        result = rag_service.query(question)
        results["question"].append(question)
        results["answer"].append(result["answer"])
        results["contexts"].append([c for c in result.get("contexts", [])])
        results["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(results)

    # 執行 RAGAS 評估（使用 LLM 作為評委）
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    return {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_recall": round(scores["context_recall"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "overall": round(
            (scores["faithfulness"] + scores["answer_relevancy"] +
             scores["context_recall"] + scores["context_precision"]) / 4, 4
        )
    }

# 週期性評估（可設為 cron job）
def run_periodic_evaluation(db: FeedbackDatabase, rag_service: FeedbackAwareRAG):
    """
    每週從反饋數據中抽樣，自動評估系統效果
    """
    # 從資料庫取得有修正答案的記錄（品質最高的評估數據）
    rows = db.conn.execute("""
        SELECT ql.query, uf.correction
        FROM query_logs ql
        JOIN user_feedback uf ON ql.query_id = uf.query_id
        WHERE uf.correction IS NOT NULL
        ORDER BY uf.timestamp DESC
        LIMIT 50
    """).fetchall()

    if not rows:
        print("沒有足夠的評估數據")
        return

    questions = [r[0] for r in rows]
    ground_truths = [r[1] for r in rows]

    scores = evaluate_rag_system(questions, ground_truths, rag_service)

    print("=== RAGAS 評估報告 ===")
    print(f"忠實度 (Faithfulness):     {scores['faithfulness']:.2%}")
    print(f"答案相關性 (Relevancy):    {scores['answer_relevancy']:.2%}")
    print(f"上下文召回率 (Recall):     {scores['context_recall']:.2%}")
    print(f"上下文精確度 (Precision):  {scores['context_precision']:.2%}")
    print(f"綜合評分:                  {scores['overall']:.2%}")

    stats = db.get_training_stats()
    print(f"\n=== 反饋數據統計 ===")
    print(f"累計反饋筆數：{stats['total_feedback']}")
    print(f"訓練數據對數：{stats['training_pairs']}")
    print(f"平均評分：{stats['average_rating']}/5")
    print(f"快取大小：{stats['cache_size']} 筆")

    return scores
```

### Embedding Fine-tuning 數據準備

```python
def export_finetuning_dataset(db: FeedbackDatabase, output_path: str):
    """
    從反饋資料庫匯出 Embedding fine-tuning 所需的三元組訓練數據
    格式：(query, positive_chunk, negative_chunk)
    
    可直接用於 sentence-transformers 的 TripletLoss 訓練
    """
    rows = db.conn.execute("""
        SELECT tp.query, tp.positive_chunk_id, tp.negative_chunk_id
        FROM training_pairs tp
        WHERE tp.source = 'explicit'
        ORDER BY tp.created_at DESC
        LIMIT 10000
    """).fetchall()

    dataset = []
    for query, pos_id, neg_id in rows:
        # 實際應用中，需要從文件庫根據 chunk_id 取得原文
        dataset.append({
            "query": query,
            "positive_id": pos_id,
            "negative_id": neg_id
        })

    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"匯出 {len(dataset)} 筆訓練數據到 {output_path}")
    print("可使用 sentence-transformers TripletLoss 進行 fine-tuning")
    print("預期效果：domain-specific 查詢的相關性提升 5-15%")
```

## 優缺點分析

### 優點

**1. 系統持續進化，越用越準**
初期部署的 RAG 系統通常效果有限，但隨著反饋數據積累，chunk 權重、快取和 embedding 模型都在改善，形成正向飛輪。

**2. 快取大幅降低成本**
企業知識庫中，相同或相似問題的重複率通常超過 30%。高品質答案快取後直接返回，無需重跑 RAG 管道，成本降為零。

**3. 識別系統薄弱點**
通過分析點踩率高的查詢類型，可以快速定位哪類問題 RAG 回答得差，針對性優化（補充文件、調整 chunking 策略）。

**4. 解決 Domain Shift 問題**
業務變化導致查詢分佈改變時，反饋數據可以快速反映，fine-tuning 後的 embedding 模型能適應新的查詢模式。

### 缺點

**1. 冷啟動問題**
新系統沒有反饋數據，所有改善路徑都無法啟動。需要設計種子數據（如人工標注 100-500 對）或透過其他方式引導初期用戶留下反饋。

**2. 反饋偏差（Selection Bias）**
只有少數用戶會主動留下反饋（通常 < 5%），且點踩（負面體驗）的轉化率往往高於點讚，導致訓練數據偏向負面。

**3. Fine-tuning 週期長**
累積足夠的訓練數據（通常需要 1000+ 三元組）、執行 fine-tuning、部署新模型需要數天到數週，無法即時改善。

**4. 用戶反饋可能有噪音**
用戶的「有幫助」可能是因為回答符合既有偏見，而非真的正確。需要設計去噪機制（如多數決、置信度加權）。

## 適用場景

**最適合建立 Feedback Loop 的場景：**

1. **企業內部知識庫**：員工每天使用，反饋量大且動機強（遇到錯誤回答直接影響工作）
2. **長期運營的問答系統**：能夠積累數月甚至數年的反饋數據
3. **垂直領域 AI 助理**：法律、醫療、金融等領域，通用 embedding 模型適配性差，fine-tuning 效益顯著
4. **客服機器人**：有明確的「問題解決率」指標，反饋信號清晰

**不適合的場景：**
- 一次性或短期專案（沒有足夠時間積累反饋）
- 用戶數量極少（< 100 人/天）的系統
- 高度隱私的場景（不能記錄用戶互動）

## 與其他方法的比較

| 方法 | 改善速度 | 效果上限 | 維護成本 | 冷啟動 |
|------|---------|---------|---------|-------|
| 靜態 RAG（無反饋） | 無 | 低 | 低 | 無問題 |
| Chunk 權重更新 | 即時 | 中 | 低 | 可用 |
| Q&A 快取 | 即時 | 中（只解決重複問題） | 低 | 可用 |
| RAGAS 週期評估 | 週期 | 中（監控為主） | 中 | 需標注數據 |
| Embedding Fine-tuning | 慢（週級） | 高 | 高 | 需大量數據 |
| 全部整合 | 多層次 | 最高 | 高 | 困難 |

**與 Adaptive RAG 的互補**：Feedback Loop 改善各個組件的品質，Adaptive RAG 改善策略選擇的準確度——兩者可以疊加使用。

## 小結

Feedback Loop 是將 RAG 從「靜態系統」轉變為「持續學習系統」的關鍵機制。核心原則是：**每一次用戶互動都是一個訓練信號，不應被浪費**。

實作優先順序建議：
1. **第一步**：先建立查詢記錄和反饋收集基礎設施（一週內可完成）
2. **第二步**：實作 Q&A 快取和 Chunk 權重更新（即時效果，低成本）
3. **第三步**：接入 RAGAS 自動評估，建立效果監控儀表板
4. **第四步**：當累積 1000+ 訓練對後，啟動 Embedding fine-tuning 週期

最終目標：讓系統在生產環境中每個月都能看到可量化的效果提升，而非部署後就靜止不動。
