---
title: "Adaptive RAG：根據查詢複雜度動態選擇最佳檢索策略"
description: "固定的 RAG 管道對所有查詢一視同仁：「今天幾號？」和「請分析台積電近五年的競爭策略」用的是完全相同的流程。Adaptive RAG 先分類查詢的複雜度和類型，再路由到最適合的策略，從直接回答到多步驟推理檢索，讓每個查詢得到最適當的處理。"
date: 2025-01-12
tags: ["RAG", "Adaptive RAG", "Query Classification", "Self-RAG", "Multi-hop", "LangGraph", "Router"]
---

# Adaptive RAG：根據查詢複雜度動態選擇最佳檢索策略

標準 RAG 的設計假設是「所有查詢都需要檢索」，但這個假設有明顯的問題：簡單的事實查詢被過度處理，複雜的推理查詢又因單次檢索不足而回答不完整。Adaptive RAG 在查詢入口插入一個分類器，將問題按類型路由到不同的處理管道：不需要外部知識的問題直接用 LLM 回答，簡單事實查詢走輕量 RAG，複雜推理查詢走多步驟迭代檢索。整體結果是更高的準確率和更低的平均延遲。

## 核心概念

### 固定策略的代價

標準 RAG 的問題不只是「有時候不需要檢索」，而是**不同類型的查詢需要根本不同的處理方式**：

```
查詢 A：「台灣的首都是哪裡？」
  → 適合：Direct Answer（LLM 直接回答，無需檢索）
  → 標準 RAG 的做法：向量搜尋 → 找到含「台北」的 chunk → 讓 LLM 從 chunk 中提取
  → 問題：多餘的檢索步驟浪費 300ms 和 0.01 美元

查詢 B：「台積電 2nm 製程量產的時程為何？」
  → 適合：Single Retrieval（一次檢索即可）
  → 標準 RAG：✅ 正常工作

查詢 C：「相比三星，台積電的 3nm 良率優勢是否足以支撐其高於市場的代工報價？」
  → 適合：Multi-hop Retrieval（需要台積電良率資訊 + 三星良率資訊 + 代工定價邏輯）
  → 標準 RAG 的做法：單次檢索 → 答案片面，遺漏關鍵比較維度
  → 問題：需要多輪檢索才能收集完整資訊

查詢 D：「你知道我們公司的 OKR 系統是誰設計的嗎？」
  → 適合：Conversational Answer（從對話歷史推斷，不需要檢索）
  → 標準 RAG：可能觸發無效搜尋
```

### 三個自適應維度

**維度一：Query Classifier（查詢分類器）**

將查詢分為四種類型：

| 類型 | 定義 | 例子 | 最佳策略 |
|------|------|------|---------|
| 無需檢索型 | LLM 訓練數據中已有答案 | 「Python 中 list 的排序方法？」 | Direct LLM |
| 簡單事實型 | 需要一次檢索，答案明確 | 「公司的退款政策是什麼？」 | Single RAG |
| 複雜推理型 | 需要多個知識片段組合推理 | 「A 和 B 方案哪個更適合我的情況？」 | Multi-hop RAG |
| 對話型 | 答案在對話歷史中 | 「你剛才說的第三點是什麼意思？」 | Context Window |

**維度二：Strategy Selector（策略選擇器）**

基於分類結果，動態選擇 RAG 管道：

```
Direct Answer：   Query → LLM → Response
Single RAG：      Query → Retriever → LLM → Response
Multi-hop RAG：   Query → Retriever → LLM(分析) → 補充查詢 → Retriever → LLM → Response
Conversational：  Query + History → LLM → Response
```

**維度三：Iteration Controller（迭代控制器）**

對 Multi-hop 策略，決定何時停止迭代：
- 收集到的資訊足以回答問題 → 停止
- 達到最大迭代次數（如 3 次）→ 強制停止
- LLM 判斷無需更多資訊 → 停止

## 運作原理

### 完整架構圖

```
使用者查詢
     │
     ↓
┌─────────────────┐
│  Query Analyzer  │  ← 分析查詢類型、複雜度、時效性
└────────┬────────┘
         │
    ┌────┴─────┐
    │  Router   │  ← 路由決策
    └────┬──────┘
         │
    ┌────┼──────────────────────────┐
    ↓    ↓                          ↓
┌───────┐ ┌──────────┐    ┌──────────────────┐
│Direct │ │Single RAG│    │  Multi-hop RAG   │
│Answer │ │          │    │                  │
│       │ │Retriever │    │  Iteration Loop: │
│       │ │    ↓     │    │  1. Sub-query    │
│ LLM   │ │  LLM     │    │  2. Retrieve     │
└───┬───┘ └────┬─────┘    │  3. Analyze      │
    │          │          │  4. Enough? Y/N  │
    │          │          │  5. Final Answer │
    │          │          └────────┬─────────┘
    │          │                   │
    └──────────┴───────────────────┘
                    │
              最終回答 + 來源
```

### 查詢分類的判斷依據

LLM 分類器需要考量多個因素：

```python
分類規則（優先順序）：

1. 時效性指標 → 如果查詢包含「今天」「最新」「現在」，且知識庫有即時數據 → Single/Multi RAG
2. 知識截止日期 → 如果是 LLM 訓練數據截止日期後的事件 → 需要 RAG
3. 公司專有知識 → 如果問及公司內部流程、產品規格 → 需要 RAG
4. 跨概念推理 → 如果需要結合 2 個以上獨立主題 → Multi-hop RAG
5. 通用知識 → 如果是普通常識、程式語言基礎 → Direct Answer
6. 對話指向 → 如果包含「你剛才」「上面說的」 → Conversational
```

## Python 實作範例

完整的 Adaptive RAG 系統，使用 LLM 分類器路由到不同管道：

```python
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import json

# ── 查詢類型枚舉 ──────────────────────────────────────────────────────────
class QueryType(str, Enum):
    DIRECT = "direct"         # 直接用 LLM 回答，不需要檢索
    SIMPLE = "simple"         # 簡單單次 RAG
    COMPLEX = "complex"       # 複雜多步驟 RAG
    CONVERSATIONAL = "conversational"  # 對話歷史中可回答

@dataclass
class ClassificationResult:
    query_type: QueryType
    confidence: float          # 分類信心度 0-1
    reasoning: str             # 分類理由
    sub_queries: List[str]     # 若為 complex，預先拆解的子查詢

@dataclass
class RAGResult:
    answer: str
    query_type: QueryType
    iterations: int            # 實際執行的迭代次數
    retrieved_chunks: List[Document]
    sub_queries_used: List[str]

# ── 查詢分類器 ────────────────────────────────────────────────────────────
class QueryClassifier:
    """使用 LLM 對查詢進行分類，決定最適合的處理策略"""

    CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一個 RAG 查詢路由系統。請分析用戶查詢，決定最合適的處理策略。

策略說明：
- direct：問題是通用知識，LLM 訓練數據足以回答（程式語言基礎、數學、常識等）
- simple：需要從知識庫檢索一次，問題聚焦、答案明確
- complex：需要從知識庫多次檢索，問題跨越多個主題或需要比較/推理
- conversational：問題指向對話歷史（包含「你剛才說」「上面提到」等）

回傳 JSON 格式：
{{
  "query_type": "direct|simple|complex|conversational",
  "confidence": 0.0-1.0,
  "reasoning": "分類理由（一句話）",
  "sub_queries": ["子查詢1", "子查詢2"]  // 僅 complex 類型需要填寫
}}

知識庫內容領域：{domain_description}
對話歷史（最近3輪）：{recent_history}
"""),
        ("human", "用戶查詢：{query}")
    ])

    def __init__(
        self,
        domain_description: str = "公司產品文檔、技術規範、業務流程",
        model: str = "gpt-4o-mini"
    ):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.domain_description = domain_description

    def classify(
        self,
        query: str,
        conversation_history: List[Dict] = None
    ) -> ClassificationResult:
        """
        分析查詢並返回分類結果
        """
        history_str = ""
        if conversation_history:
            recent = conversation_history[-3:]  # 只看最近 3 輪
            history_str = "\n".join([
                f"{'用戶' if m['role'] == 'user' else '助理'}：{m['content'][:100]}"
                for m in recent
            ])

        chain = self.CLASSIFICATION_PROMPT | self.llm

        try:
            response = chain.invoke({
                "domain_description": self.domain_description,
                "recent_history": history_str or "（無歷史）",
                "query": query
            })

            result = json.loads(response.content)
            return ClassificationResult(
                query_type=QueryType(result["query_type"]),
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", ""),
                sub_queries=result.get("sub_queries", [])
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            # 分類失敗時回退到 simple 策略（最安全的預設值）
            return ClassificationResult(
                query_type=QueryType.SIMPLE,
                confidence=0.5,
                reasoning="分類失敗，使用預設 simple 策略",
                sub_queries=[]
            )

# ── 各策略的執行器 ────────────────────────────────────────────────────────
class DirectAnswerExecutor:
    """直接使用 LLM 回答，不進行任何檢索"""

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", "你是一個知識助理，請直接回答問題。如果不確定，請誠實說明。"),
        ("human", "{query}")
    ])

    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def execute(self, query: str) -> RAGResult:
        chain = self.PROMPT | self.llm
        answer = chain.invoke({"query": query}).content
        return RAGResult(
            answer=answer,
            query_type=QueryType.DIRECT,
            iterations=0,
            retrieved_chunks=[],
            sub_queries_used=[]
        )


class SimpleRAGExecutor:
    """標準單次 RAG 流程"""

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """根據以下檢索到的內容回答問題。
若內容不足以回答，請說明缺少哪些資訊。

檢索內容：
{context}
"""),
        ("human", "{query}")
    ])

    def __init__(self, retriever, model: str = "gpt-4o"):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model, temperature=0)

    def execute(self, query: str) -> RAGResult:
        # 單次檢索
        chunks = self.retriever.get_relevant_documents(query)
        context = "\n\n---\n\n".join(c.page_content for c in chunks[:4])

        chain = self.PROMPT | self.llm
        answer = chain.invoke({"context": context, "query": query}).content

        return RAGResult(
            answer=answer,
            query_type=QueryType.SIMPLE,
            iterations=1,
            retrieved_chunks=chunks,
            sub_queries_used=[query]
        )


class MultiHopRAGExecutor:
    """多步驟迭代 RAG 流程"""

    # 子查詢生成的 Prompt
    SUBQUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你正在協助回答一個複雜問題。
已收集到的資訊：
{collected_info}

原始問題：{original_query}

判斷是否已有足夠資訊回答問題。
- 若足夠：回傳 {{"sufficient": true, "next_query": null}}
- 若不足：回傳 {{"sufficient": false, "next_query": "下一個需要搜尋的具體問題"}}

注意：next_query 應該是一個具體的、可搜尋的問題，不是原始問題的重複。"""),
        ("human", "請判斷是否需要繼續搜尋，並提供下一個查詢（若需要）。")
    ])

    SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """根據以下多輪檢索收集到的資訊，綜合回答原始問題。

收集到的資訊（按檢索順序）：
{all_context}

原始問題：{original_query}

請提供完整、有條理的回答，可以使用列表或段落組織。
"""),
        ("human", "請綜合以上資訊回答問題。")
    ])

    def __init__(
        self,
        retriever,
        model: str = "gpt-4o",
        max_iterations: int = 3
    ):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.max_iterations = max_iterations

    def execute(
        self,
        query: str,
        initial_sub_queries: List[str] = None
    ) -> RAGResult:
        """
        迭代執行多步驟檢索
        """
        all_chunks = []
        all_sub_queries = []
        collected_contexts = []

        # 第一輪：使用預先生成的子查詢（若有）或原始查詢
        current_queries = initial_sub_queries if initial_sub_queries else [query]
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # 執行當前輪的查詢（若有多個子查詢，並行搜尋）
            round_chunks = []
            for sub_q in current_queries:
                chunks = self.retriever.get_relevant_documents(sub_q)
                round_chunks.extend(chunks[:2])  # 每個子查詢取 top-2
                all_sub_queries.append(sub_q)

            # 去重（相同 chunk_id 只保留一次）
            seen_ids = set()
            for chunk in round_chunks:
                chunk_id = chunk.metadata.get("chunk_id", chunk.page_content[:50])
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_chunks.append(chunk)
                    collected_contexts.append(
                        f"[子查詢 {iteration}：{current_queries[0]}]\n{chunk.page_content}"
                    )

            # 判斷是否已有足夠資訊
            if iteration >= self.max_iterations:
                break  # 達到最大迭代次數

            chain = self.SUBQUERY_PROMPT | self.llm
            decision = chain.invoke({
                "collected_info": "\n\n".join(collected_contexts[-4:]),  # 只看最近4筆
                "original_query": query
            })

            try:
                decision_data = json.loads(decision.content)
                if decision_data.get("sufficient", False):
                    break  # LLM 判斷資訊足夠
                next_query = decision_data.get("next_query")
                if not next_query:
                    break
                current_queries = [next_query]  # 下一輪的查詢
            except json.JSONDecodeError:
                break  # 解析失敗時停止

        # 最終綜合回答
        chain = self.SYNTHESIS_PROMPT | self.llm
        answer = chain.invoke({
            "all_context": "\n\n".join(collected_contexts),
            "original_query": query
        }).content

        return RAGResult(
            answer=answer,
            query_type=QueryType.COMPLEX,
            iterations=iteration,
            retrieved_chunks=all_chunks,
            sub_queries_used=all_sub_queries
        )


class ConversationalExecutor:
    """從對話歷史中回答，不進行外部檢索"""

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", "你是一個對話助理。根據對話歷史回答用戶的問題。"),
        ("human", "對話歷史：\n{history}\n\n當前問題：{query}")
    ])

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def execute(self, query: str, history: List[Dict]) -> RAGResult:
        history_str = "\n".join([
            f"{'用戶' if m['role'] == 'user' else '助理'}：{m['content']}"
            for m in history[-6:]  # 最近 6 輪對話
        ])

        chain = self.PROMPT | self.llm
        answer = chain.invoke({"history": history_str, "query": query}).content

        return RAGResult(
            answer=answer,
            query_type=QueryType.CONVERSATIONAL,
            iterations=0,
            retrieved_chunks=[],
            sub_queries_used=[]
        )

# ── Adaptive RAG 主控制器 ─────────────────────────────────────────────────
class AdaptiveRAG:
    """
    自適應 RAG 系統：根據查詢類型動態路由到最適合的處理管道
    """

    def __init__(
        self,
        retriever,
        domain_description: str = "公司知識庫",
        classifier_model: str = "gpt-4o-mini",
        generator_model: str = "gpt-4o",
        max_iterations: int = 3
    ):
        # 初始化分類器
        self.classifier = QueryClassifier(
            domain_description=domain_description,
            model=classifier_model
        )

        # 初始化各策略執行器
        self.executors = {
            QueryType.DIRECT: DirectAnswerExecutor(model=generator_model),
            QueryType.SIMPLE: SimpleRAGExecutor(retriever, model=generator_model),
            QueryType.COMPLEX: MultiHopRAGExecutor(
                retriever, model=generator_model, max_iterations=max_iterations
            ),
            QueryType.CONVERSATIONAL: ConversationalExecutor(model=classifier_model),
        }

        self.conversation_history: List[Dict] = []

    def query(self, user_query: str, verbose: bool = False) -> Dict:
        """
        主查詢入口：自動分類並路由

        Args:
            user_query: 用戶的問題
            verbose: 是否輸出詳細的路由資訊

        Returns:
            包含答案、策略資訊的字典
        """
        # 1. 查詢分類
        classification = self.classifier.classify(
            user_query,
            conversation_history=self.conversation_history
        )

        if verbose:
            print(f"[路由] 查詢類型：{classification.query_type.value}")
            print(f"[路由] 信心度：{classification.confidence:.2%}")
            print(f"[路由] 理由：{classification.reasoning}")

        # 2. 執行對應策略
        executor = self.executors[classification.query_type]

        if classification.query_type == QueryType.CONVERSATIONAL:
            result = executor.execute(user_query, self.conversation_history)
        elif classification.query_type == QueryType.COMPLEX and classification.sub_queries:
            result = executor.execute(user_query, classification.sub_queries)
        else:
            result = executor.execute(user_query)

        # 3. 更新對話歷史
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": result.answer})

        if verbose:
            print(f"[執行] 迭代次數：{result.iterations}")
            print(f"[執行] 使用子查詢：{result.sub_queries_used}")

        return {
            "answer": result.answer,
            "query_type": result.query_type.value,
            "strategy_info": {
                "classification_confidence": classification.confidence,
                "classification_reasoning": classification.reasoning,
                "iterations": result.iterations,
                "sub_queries": result.sub_queries_used,
                "chunks_retrieved": len(result.retrieved_chunks),
            }
        }

# ── 使用範例 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 建立知識庫
    from langchain.schema import Document as LCDocument

    docs = [
        LCDocument(page_content="台積電 2nm 製程預計 2025 年量產，採用 GAA 電晶體架構。",
                   metadata={"chunk_id": "tsmc_2nm"}),
        LCDocument(page_content="台積電 3nm 製程良率已達 80% 以上，三星 3nm 良率約 60%。",
                   metadata={"chunk_id": "tsmc_yield"}),
        LCDocument(page_content="晶圓代工報價：台積電 3nm 每片晶圓約 2 萬美元，三星約 1.5 萬美元。",
                   metadata={"chunk_id": "wafer_price"}),
        LCDocument(page_content="公司差旅報銷需在出差後 7 個工作日內提交申請。",
                   metadata={"chunk_id": "expense_policy"}),
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 建立 Adaptive RAG 系統
    adaptive_rag = AdaptiveRAG(
        retriever=retriever,
        domain_description="半導體技術文檔和公司內部政策",
        max_iterations=3
    )

    # 測試不同類型的查詢
    test_queries = [
        ("Python 中 enumerate 函數的用法？", "→ 預期路由：direct"),
        ("差旅費報銷的申請期限？", "→ 預期路由：simple"),
        ("台積電的良率優勢是否支撐其高於三星的報價？", "→ 預期路由：complex"),
    ]

    for query, expected in test_queries:
        print(f"\n{'='*60}")
        print(f"查詢：{query}")
        print(f"預期：{expected}")
        result = adaptive_rag.query(query, verbose=True)
        print(f"\n回答：{result['answer'][:200]}...")
```

### 效能對比分析

```python
import time

def benchmark_strategies(adaptive_rag: AdaptiveRAG, test_cases: List[Dict]) -> None:
    """
    對比 Adaptive RAG 與固定策略的效能差異
    
    test_cases 格式：
    [{"query": "...", "true_type": "direct|simple|complex", "expected_answer_keywords": [...]}]
    """
    results = {
        "adaptive": {"latency": [], "accuracy": [], "cost_tokens": []},
        "fixed_rag": {"latency": [], "accuracy": [], "cost_tokens": []},
    }

    print(f"{'查詢':<40} {'真實類型':<12} {'Adaptive路由':<12} {'延遲節省':>10}")
    print("-" * 80)

    for case in test_cases:
        query = case["query"]
        true_type = case["true_type"]

        # Adaptive RAG 測試
        start = time.time()
        adaptive_result = adaptive_rag.query(query)
        adaptive_latency = (time.time() - start) * 1000

        routed_type = adaptive_result["query_type"]
        is_correct_route = (routed_type == true_type)

        # 模擬固定 RAG 的延遲（假設所有查詢都走 simple RAG）
        fixed_latency = adaptive_latency * (1.5 if true_type == "direct" else 1.0)

        latency_saved = fixed_latency - adaptive_latency
        results["adaptive"]["latency"].append(adaptive_latency)
        results["fixed_rag"]["latency"].append(fixed_latency)

        route_display = f"{'✅' if is_correct_route else '❌'} {routed_type}"
        print(f"{query[:38]:<40} {true_type:<12} {route_display:<12} {latency_saved:>+8.0f}ms")

    avg_adaptive = sum(results["adaptive"]["latency"]) / len(results["adaptive"]["latency"])
    avg_fixed = sum(results["fixed_rag"]["latency"]) / len(results["fixed_rag"]["latency"])

    print(f"\n平均延遲比較：")
    print(f"  Adaptive RAG：{avg_adaptive:.0f}ms")
    print(f"  固定 RAG：    {avg_fixed:.0f}ms")
    print(f"  節省：        {avg_fixed - avg_adaptive:.0f}ms ({(1 - avg_adaptive/avg_fixed)*100:.1f}%)")
```

## 優缺點分析

### 優點

**1. 對簡單查詢效率大幅提升**
在典型的企業知識庫場景中，「直接回答型」查詢通常佔 20-30%（如常識問題、程式語言問題）。這些查詢完全跳過了檢索步驟，延遲從 1500ms 降到 300ms，成本降低 80%。

**2. 對複雜查詢效果顯著改善**
Multi-hop 策略允許系統收集多方面資訊再綜合回答，對比較分析類問題（A vs B）、需要多個事實組合的問題，準確率提升 15-30%。

**3. 系統更智慧、更自然**
不再對所有問題一律「先搜尋再回答」，行為更像一個有判斷力的人類助理。

**4. 可解釋性強**
每個查詢都有明確的路由記錄（類型、信心度、理由），方便排查錯誤路由。

### 缺點

**1. 分類本身有成本**
LLM 分類器需要一次 LLM 調用，約增加 200-500ms 延遲和少量 token 成本（可用 gpt-4o-mini 降低成本）。

**2. 分類錯誤的代價**
若把 complex 查詢錯誤路由到 direct（直接回答），答案可能完全錯誤。需要設計 fallback 機制和信心度閾值（低於 0.7 時回退到 simple）。

**3. Multi-hop 延遲高**
3 次迭代可能累計 4-6 秒延遲，在即時對話場景中用戶體驗差。需要設定嚴格的迭代上限。

**4. 需要精心設計分類 Prompt**
分類效果高度依賴 Prompt 品質，且隨著知識庫內容變化可能需要調整。

## 適用場景

**最適合 Adaptive RAG 的場景：**

| 場景 | 理由 |
|------|------|
| 通用型問答助理 | 查詢類型高度多樣，固定策略無法兼顧 |
| 多功能聊天機器人 | 需要在對話、檢索、推理之間靈活切換 |
| 複雜分析類應用 | 競品分析、法律文件比較等需要 multi-hop |
| 高流量系統 | Direct Answer 為大量簡單查詢節省成本 |

**可能不需要 Adaptive RAG 的場景：**
- 查詢類型單一（如「查文件」系統，幾乎都是 simple RAG）
- 對延遲極度敏感（分類器增加 200-500ms）
- 小型專案（過度工程化）

## 與其他方法的比較

| 方法 | 查詢分類 | 多步推理 | 實作複雜度 | 延遲 |
|------|---------|---------|-----------|------|
| 標準 RAG | ❌ | ❌ | 低 | 低 |
| Self-RAG | 部分（自評估） | 部分（迭代） | 中 | 中 |
| Adaptive RAG | ✅ 明確分類 | ✅ 結構化迭代 | 中高 | 中 |
| FLARE | ❌ | ✅（按需檢索） | 中 | 中 |
| GraphRAG | ❌ | ✅（圖遍歷） | 高 | 高 |

### 與 Self-RAG 的關係

Self-RAG（Self-Reflective RAG）讓 LLM 自己決定何時需要檢索、檢索結果是否有用，並對輸出品質自我評分。Adaptive RAG 是更結構化的版本：

```
Self-RAG：LLM 在生成過程中自主決定是否檢索（隱式路由）
Adaptive RAG：在查詢入口顯式分類，路由到不同管道（顯式路由）

Adaptive RAG 的優勢：
- 可預測性更高（路由邏輯明確）
- 除錯更容易（有分類記錄）
- 各策略可以獨立優化

Self-RAG 的優勢：
- 更靈活（LLM 可在生成中途決定檢索）
- 對 prompt 設計要求更低
```

### 使用 LangGraph 實作更複雜的路由

```python
# LangGraph 提供更強大的有向圖工作流，適合複雜的 Adaptive RAG
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    query_type: str
    retrieved_docs: list
    answer: str
    iteration: int

def build_adaptive_rag_graph(retriever, llm):
    """使用 LangGraph 建立可視化的 Adaptive RAG 工作流"""
    workflow = StateGraph(AgentState)

    # 添加節點
    workflow.add_node("classify", classify_query_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("check_sufficient", check_sufficient_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # 設定起點
    workflow.set_entry_point("classify")

    # 設定條件路由
    workflow.add_conditional_edges(
        "classify",
        route_by_type,  # 路由函數
        {
            "direct": "direct_answer",
            "simple": "retrieve",
            "complex": "retrieve",
            "conversational": "direct_answer"
        }
    )

    # 迭代控制
    workflow.add_conditional_edges(
        "check_sufficient",
        should_continue,
        {"continue": "retrieve", "done": "generate_answer"}
    )

    workflow.add_edge("retrieve", "check_sufficient")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()
```

## 小結

Adaptive RAG 的核心洞察是：**不同查詢的最佳處理策略差異巨大，用單一固定管道是一種浪費**。透過在查詢入口加入分類路由層，系統能夠：

- 簡單問題快速回答（節省 70% 延遲和成本）
- 複雜問題深度檢索（提升 15-30% 準確率）
- 對話問題流暢銜接（不打斷對話節奏）

**實作建議：**

1. **從簡單開始**：先用規則判斷（regex 檢查對話詞、關鍵字數量），不必一開始就用 LLM 分類器
2. **監控路由準確率**：記錄每次路由決策，定期人工抽查是否路由正確
3. **設定安全回退**：分類信心度 < 0.7 時，一律回退到 simple RAG（安全的通用策略）
4. **逐步升級**：從 Direct + Simple 兩種策略開始，等有足夠使用數據後再加入 Complex 策略

Adaptive RAG 不是萬靈藥，但對於需要同時處理多樣化查詢的通用助理系統，它是目前最有效的架構之一。
