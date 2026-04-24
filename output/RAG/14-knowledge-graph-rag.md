---
title: "Knowledge Graph RAG：用知識圖譜解鎖關係推理能力"
description: "向量搜尋擅長語義相似度，但無法推理「A 的母公司的 CEO 的兄弟是誰」這類多跳關係問題。Knowledge Graph RAG 結合知識圖譜的結構化關係與 LLM 的語言生成，讓 RAG 系統具備真正的關係推理能力。"
date: 2024-01-14
tags: ["RAG", "Knowledge Graph", "GraphRAG", "Neo4j", "知識圖譜", "關係推理"]
---

# Knowledge Graph RAG：用知識圖譜解鎖關係推理能力

知識圖譜增強 RAG（Knowledge Graph RAG）是為了解決向量搜尋的根本局限而生的。純向量 RAG 在「找到語義相似的文本片段」方面表現出色，但面對需要多步關係推理的問題——例如實體間的複雜關聯——它完全束手無策。KG RAG 透過將結構化的圖譜知識與 LLM 的語言能力結合，開啟了 RAG 系統在複雜推理任務上的新可能。

---

## 核心概念

### 向量搜尋的盲點

考慮這個問題：「台積電主要客戶的 AI 晶片部門負責人曾在哪所大學就讀？」

這個問題涉及多個推理步驟：
1. 台積電的主要客戶是誰？（Apple, NVIDIA, AMD...）
2. 這些客戶的 AI 晶片部門負責人是誰？
3. 這個人的教育背景？

向量搜尋無法處理這類「多跳推理」（Multi-hop Reasoning），因為每個步驟的答案都是下一步搜尋的輸入，而向量空間無法表達這種動態的推理鏈。

### 知識圖譜基礎

知識圖譜（Knowledge Graph, KG）以**三元組（Triple）**的形式儲存知識：

```
（主體 Subject, 關係 Relation, 客體 Object）
（Subject, Predicate, Object）— SPO 格式

範例：
（台積電, 創辦人, 張忠謀）
（張忠謀, 畢業於, MIT）
（台積電, 主要客戶, Apple）
（Apple, 旗下部門, Apple Silicon）
```

三個核心元素：
- **實體（Entity）**：現實世界的物件（公司、人物、地點、概念）
- **關係（Relation）**：實體間的語義連接（創辦人、客戶、位於）
- **三元組（Triple）**：一條知識的最小單位

有了圖結構，多跳推理就變成了圖上的路徑查詢：

```
台積電 ──[主要客戶]──▶ Apple ──[旗下部門]──▶ Apple Silicon
```

---

## 運作原理

### 兩種整合方式

#### 方式一：KG 輔助向量 RAG（KG-Augmented RAG）

用知識圖譜「豐富」查詢，再進行向量搜尋：

```
用戶查詢
    │
    ▼
實體識別（NER）
從查詢中提取關鍵實體
    │
    ▼
圖譜擴展查詢
在 KG 中找出相關實體和關係
    │
    ▼
構建增強查詢
原始查詢 + 圖譜上下文
    │
    ▼
向量搜尋
用增強查詢搜尋文本向量庫
    │
    ▼
LLM 生成最終答案
```

#### 方式二：KG 作為主要知識來源（GraphRAG）

Microsoft 在 2024 年提出的 GraphRAG：

```
文件集合
    │
    ▼
LLM 萃取實體和關係
構建知識圖譜
    │
    ▼
社群偵測（Community Detection）
將相關實體分組
    │
    ▼
生成社群摘要
每個社群用 LLM 生成摘要
    │
    ▼
查詢時：
├── Global 查詢：聚合所有社群摘要
└── Local 查詢：找相關子圖，再向量搜尋
```

### Microsoft GraphRAG 深度解析

GraphRAG 的獨特之處在於它的**兩層知識結構**：

**底層：細粒度知識圖譜**
從原始文件中抽取所有實體和關係，形成密集的圖結構。

**上層：社群摘要（Community Summary）**
使用 Leiden 演算法進行社群偵測，將緊密相關的實體分組，然後用 LLM 為每個社群生成自然語言摘要。

**查詢模式對比**：

| 查詢模式 | 適用問題類型 | 實作方式 |
|---------|------------|---------|
| Global Search | 「整體趨勢是什麼？」「主要主題有哪些？」| Map-Reduce：對所有社群摘要并行查詢後聚合 |
| Local Search | 「特定實體的詳細資訊」| 找相關子圖 + 向量搜尋相關文本 |

---

## Python 實作範例

### 環境準備

```bash
pip install langchain langchain-community langchain-openai neo4j
```

### 完整實作：Neo4j + LangChain GraphCypherQAChain

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────
# 步驟 1：連接 Neo4j 知識圖譜
# ─────────────────────────────────────
# 需要先啟動 Neo4j（可用 Docker：docker run -p 7474:7474 -p 7687:7687 neo4j）
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# ─────────────────────────────────────
# 步驟 2：建立範例知識圖譜
# ─────────────────────────────────────
def create_sample_kg():
    """建立台灣科技公司知識圖譜（範例資料）"""
    
    cypher_statements = [
        # 建立公司節點
        "MERGE (tsmc:Company {name: '台積電', founded: 1987, country: '台灣'})",
        "MERGE (apple:Company {name: 'Apple', founded: 1976, country: '美國'})",
        "MERGE (nvidia:Company {name: 'NVIDIA', founded: 1993, country: '美國'})",
        "MERGE (asml:Company {name: 'ASML', founded: 1984, country: '荷蘭'})",
        
        # 建立人物節點
        "MERGE (morris:Person {name: '張忠謀', title: '創辦人'})",
        "MERGE (cc_wei:Person {name: '魏哲家', title: '現任CEO'})",
        "MERGE (jensen:Person {name: 'Jensen Huang', title: 'CEO'})",
        
        # 建立大學節點
        "MERGE (mit:University {name: 'MIT', location: '美國麻省'})",
        "MERGE (stanford:University {name: 'Stanford', location: '美國加州'})",
        "MERGE (oregon:University {name: 'Oregon State', location: '美國俄勒岡'})",
        
        # 建立製程技術節點
        "MERGE (n3:Process {name: '3nm', year: 2022})",
        "MERGE (n5:Process {name: '5nm', year: 2020})",
        
        # 建立關係
        "MATCH (tsmc:Company {name: '台積電'}), (morris:Person {name: '張忠謀'}) "
        "MERGE (morris)-[:FOUNDED]->(tsmc)",
        
        "MATCH (tsmc:Company {name: '台積電'}), (cc_wei:Person {name: '魏哲家'}) "
        "MERGE (cc_wei)-[:CEO_OF]->(tsmc)",
        
        "MATCH (morris:Person {name: '張忠謀'}), (mit:University {name: 'MIT'}) "
        "MERGE (morris)-[:GRADUATED_FROM]->(mit)",
        
        "MATCH (jensen:Person {name: 'Jensen Huang'}), (oregon:University {name: 'Oregon State'}) "
        "MERGE (jensen)-[:GRADUATED_FROM]->(oregon)",
        
        "MATCH (tsmc:Company {name: '台積電'}), (apple:Company {name: 'Apple'}) "
        "MERGE (tsmc)-[:MANUFACTURES_FOR]->(apple)",
        
        "MATCH (tsmc:Company {name: '台積電'}), (nvidia:Company {name: 'NVIDIA'}) "
        "MERGE (tsmc)-[:MANUFACTURES_FOR]->(nvidia)",
        
        "MATCH (tsmc:Company {name: '台積電'}), (n3:Process {name: '3nm'}) "
        "MERGE (tsmc)-[:PRODUCES]->(n3)",
        
        "MATCH (tsmc:Company {name: '台積電'}), (asml:Company {name: 'ASML'}) "
        "MERGE (tsmc)-[:USES_EQUIPMENT_FROM]->(asml)",
        
        "MATCH (nvidia:Company {name: 'NVIDIA'}), (jensen:Person {name: 'Jensen Huang'}) "
        "MERGE (jensen)-[:CEO_OF]->(nvidia)",
    ]
    
    for stmt in cypher_statements:
        graph.query(stmt)
    
    print("知識圖譜建立完成！")
    print(f"圖譜結構：{graph.schema}")

create_sample_kg()

# ─────────────────────────────────────
# 步驟 3：建立自訂的 Cypher 生成 Prompt
# ─────────────────────────────────────
CYPHER_GENERATION_TEMPLATE = """
你是一個 Neo4j 專家。根據圖譜的 Schema 和用戶問題，生成 Cypher 查詢語句。

Schema 資訊：
{schema}

注意事項：
1. 使用 MATCH 子句尋找節點和關係
2. 使用參數化查詢防止注入
3. 對於多跳查詢，使用多個 MATCH 子句
4. 只返回 Cypher 查詢，不要解釋

用戶問題：{question}

Cypher 查詢：
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

# ─────────────────────────────────────
# 步驟 4：建立 GraphCypherQAChain
# ─────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    verbose=True,          # 顯示生成的 Cypher 語句
    return_intermediate_steps=True,  # 返回查詢過程
    allow_dangerous_requests=True    # 允許寫入操作（生產環境謹慎使用）
)

# ─────────────────────────────────────
# 步驟 5：執行多跳推理查詢
# ─────────────────────────────────────
def kg_rag_query(question: str) -> dict:
    """
    執行 KG RAG 查詢，支援多跳關係推理。
    """
    print(f"\n{'='*60}")
    print(f"問題：{question}")
    print('='*60)
    
    result = chain.invoke({"query": question})
    
    print(f"\n中間步驟（Cypher 查詢）：")
    if result.get("intermediate_steps"):
        for step in result["intermediate_steps"]:
            if "query" in step:
                print(f"  Cypher: {step['query']}")
            if "context" in step:
                print(f"  結果: {step['context']}")
    
    print(f"\n最終答案：{result['result']}")
    return result

# 測試多跳查詢
# 1. 單跳查詢
result1 = kg_rag_query("台積電的創辦人是誰？")

# 2. 兩跳查詢
result2 = kg_rag_query("台積電的創辦人從哪所大學畢業？")

# 3. 多跳複雜查詢
result3 = kg_rag_query("台積電的客戶公司的 CEO 分別在哪裡讀書？")

# 4. 聚合查詢
result4 = kg_rag_query("台積電為哪些公司生產晶片？")


# ─────────────────────────────────────
# 進階：混合 KG + 向量搜尋
# ─────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def hybrid_kg_vector_rag(question: str) -> str:
    """
    混合檢索：先用 KG 獲取結構化關係，再用向量搜尋獲取詳細文本。
    """
    # 第一步：從 KG 獲取相關實體
    entity_query = f"""
    MATCH (n) WHERE n.name CONTAINS '{question.split('的')[0] if '的' in question else question[:4]}'
    RETURN n.name as entity, labels(n)[0] as type LIMIT 5
    """
    
    try:
        kg_results = graph.query(entity_query)
        entities = [r['entity'] for r in kg_results if r['entity']]
    except Exception:
        entities = []
    
    # 第二步：用 KG 實體豐富查詢，進行向量搜尋
    sample_docs = [
        Document(page_content="台積電在3nm製程技術上取得重大突破，客戶包括Apple和NVIDIA，用於製造M系列晶片和H100 GPU。"),
        Document(page_content="ASML的EUV光刻機是台積電生產先進製程的關鍵設備，每台造價超過1億美元。"),
    ]
    
    vectorstore = Chroma.from_documents(sample_docs, OpenAIEmbeddings())
    
    # 用增強查詢進行搜尋
    enriched_query = question
    if entities:
        enriched_query = f"{question} (相關實體：{', '.join(entities)})"
    
    vector_results = vectorstore.similarity_search(enriched_query, k=2)
    
    # 組合 KG 關係 + 文本上下文
    context = f"知識圖譜關係：{kg_results}\n\n相關文本：{[d.page_content for d in vector_results]}"
    
    response = llm.invoke(f"根據以下資訊回答問題：\n{context}\n\n問題：{question}")
    return response.content

result_hybrid = hybrid_kg_vector_rag("台積電的製程技術和客戶關係？")
```

---

## 優缺點分析

### 優點

**1. 多跳推理能力**
可以沿著圖上的路徑進行推理，解決向量搜尋無法處理的複雜關係問題。

**2. 精確的關係表達**
「A 是 B 的 CEO」和「A 曾在 B 工作」是完全不同的關係，KG 能精確區分，向量空間則可能混淆。

**3. 可更新性**
知識圖譜可以精確地更新某個三元組，而不影響其他知識，比重建向量庫更精細。

**4. 可解釋的推理路徑**
Cypher 查詢提供了清晰的推理鏈，使系統決策完全可解釋。

### 缺點

**1. 構建成本高**
從非結構化文本抽取高品質的實體和關係需要大量工作，自動化方法仍有錯誤率。

**2. 關係設計需要領域專業**
圖譜的 Schema 設計需要深入理解業務領域，設計不當會嚴重影響查詢效果。

**3. 難以處理模糊知識**
KG 適合表達明確的事實，對於「A 和 B 在某種程度上相似」這類模糊關係，表達能力有限。

**4. 維護成本**
隨著知識更新，需要持續維護圖譜的一致性，包括處理衝突的三元組。

---

## 適用場景

### 最佳適用場景

**企業知識管理**
公司組織架構、產品關係、客戶關係網絡天然適合圖譜表達，可支援「誰負責哪個客戶的哪個專案」這類複雜查詢。

**生醫研究**
藥物-靶點-疾病關係、基因互動網絡、蛋白質功能關聯，是知識圖譜的經典應用場景。

**法律條文分析**
法條間的引用關係、案例先例、法律主體間的權利義務關係，適合以圖結構建模。

**金融投資分析**
上市公司的股權結構、董事交叉持股、供應鏈關係分析，需要多跳關係查詢。

### 不適合的場景

- 非結構化文本為主、關係不明確的知識庫
- 快速原型或小型專案（建構成本過高）
- 知識更新頻率極高的場景（維護成本高）

---

## 與其他方法的比較

| 特性 | 純向量 RAG | KG RAG | GraphRAG (Microsoft) |
|------|-----------|--------|---------------------|
| 多跳推理 | ✗ | ✓ | 部分支援 |
| 建構成本 | 低 | 高 | 中（自動從文本建構）|
| 語義模糊查詢 | ✓ | ✗ | ✓ |
| 可解釋性 | 低 | 高 | 中 |
| 全局摘要能力 | ✗ | ✗ | ✓ |
| 維護難度 | 低 | 高 | 中 |

**KG RAG vs GraphRAG**：
- **KG RAG** 需要人工設計 Schema 並維護，但品質更高、更精確
- **GraphRAG** 自動從文本建構圖譜，建構成本低，但品質依賴 LLM 萃取能力，且主要用於社群級別的摘要查詢而非精確關係查詢

---

## 小結

Knowledge Graph RAG 填補了純向量搜尋在結構化關係推理上的空白。透過將知識圖譜的精確關係表達與 LLM 的自然語言理解結合，系統獲得了處理複雜多跳問題的能力。

核心取捨在於：**建構和維護知識圖譜需要大量投資，但換來的是前所未有的推理深度和可解釋性**。對於企業級知識管理、生醫研究等關係密集型應用，這個投資是值得的；對於一般的文件問答，純向量方案更務實。

Microsoft GraphRAG 提供了一個折中方案：自動從文本建構圖譜，降低人工成本，雖然損失了一定的精確性，但大幅降低了採用門檻。隨著 LLM 在知識萃取上的能力不斷提升，自動構建高品質知識圖譜的可行性將越來越高。
