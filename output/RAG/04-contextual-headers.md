---
title: "Contextual Headers RAG：用標題上下文讓 Chunk 不再迷路"
description: "深入解析 Contextual Headers RAG 的兩種實作方法——標題路徑前置與 Anthropic Contextual Retrieval，包含完整 Python 範例與成本分析。"
date: 2025-01-18
tags:
  - RAG
  - LLM
  - Contextual Retrieval
  - Vector Search
  - Anthropic
  - LangChain
---

# Contextual Headers RAG：用標題上下文讓 Chunk 不再迷路

一個技術手冊的 chunk 被切出來後，內容是這樣的：

> 「預設值為 `true`，可通過 `config.yml` 修改。」

這段文字對 retriever 來說幾乎沒有意義——它是哪個功能的設定？屬於哪個模組？影響什麼行為？向量模型無從判斷，只能靠字面語義去比對，結果就是使用者問「JWT 要怎麼關掉？」時，這個 chunk 可能根本不會被召回。

加上標題路徑之後，同一段內容變成：

> 「第三章 > 3.2 認證設定 > JWT Token > 預設值為 `true`，可通過 `config.yml` 修改。」

語意完全不同。Contextual Headers RAG 就是在解決這個問題：**chunk 被抽離原始文件後，結構資訊隨之消失**，導致 retrieval 品質下降。

---

## 核心概念

傳統 RAG pipeline 把文件切成固定大小的 chunk，再對每個 chunk 做向量化。這個做法有一個根本性的缺陷：**chunk 是孤立的**。原始文件中的章節關係、段落隸屬、標題層級，在切割之後全部消失。

Contextual Headers RAG 的核心想法是：在 chunk 進入向量索引之前，先替它補上「它從哪裡來」的資訊。補充方式有兩種：

1. **標題路徑前置（Header Path Prepending）**：解析文件的標題層級（H1 → H2 → H3），在 chunk 前面加上 breadcrumb 式的路徑字串。適合結構清晰的 Markdown、HTML 文件。

2. **Anthropic Contextual Retrieval**：2024 年 9 月，Anthropic 發表了一種用 LLM 自動為每個 chunk 生成上下文說明的方法。每段說明不超過 100 tokens，格式是 `[context]\n\n[original chunk]`。Anthropic 報告顯示，這個方法讓 retrieval 失敗率從 **5.7% 降至 2.9%**（降幅 49%）；若再搭配 BM25 hybrid search，可進一步降至 **1.9%**。

兩種方法可以單獨使用，也可以組合——先用標題路徑提供結構資訊，再用 LLM 補充語意說明。

---

## 運作原理

### 方法一：標題路徑前置

```
原始 Markdown 文件
        │
        ▼
┌───────────────────┐
│  解析標題層級      │  re.findall / markdown parser
│  H1 > H2 > H3    │
└────────┬──────────┘
         │  維護「當前標題路徑」狀態
         ▼
┌───────────────────┐
│  切割成 chunks     │  MarkdownHeaderTextSplitter
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────┐
│  前置標題路徑到每個 chunk        │
│  "H1 > H2 > H3\n\n{chunk}"    │
└────────┬────────────────────────┘
         │
         ▼
┌───────────────────┐
│  向量化 + 建索引  │  embedding model
└───────────────────┘
```

### 方法二：Anthropic Contextual Retrieval

```
原始文件（完整）
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
┌──────────────┐             ┌──────────────────┐
│  切割 chunks │             │  保留完整文件     │
└──────┬───────┘             │  作為 LLM 輸入   │
       │                     └────────┬─────────┘
       │  for each chunk              │
       ▼                              │
┌──────────────────────────────────────────────┐
│  LLM（Claude / GPT）                          │
│  輸入：完整文件 + 目標 chunk                   │
│  輸出：100 tokens 以內的上下文說明             │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
             ┌────────────────────┐
             │  context + chunk   │  拼接
             └────────┬───────────┘
                      │
                      ▼
             ┌────────────────────┐
             │  向量化 + 建索引   │
             └────────────────────┘
             （可選：只對原始 chunk 建索引，
               context 僅用於提升向量品質）
```

**關鍵設計決策**：向量索引時，可以只對「原始 chunk」做索引（context 只影響 embedding 輸入），或對「context + chunk」整體做索引。Anthropic 建議對整體做索引，因為這樣 context 的語意也會被編碼進向量。

---

## Python 實作範例

```python
"""
Contextual Headers RAG 完整實作
涵蓋：標題路徑前置 + Anthropic Contextual Retrieval
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import anthropic
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────

@dataclass
class ContextualChunk:
    """帶有上下文的 chunk"""
    original_content: str       # 原始 chunk 內容
    context: str                # 補充的上下文說明
    header_path: str            # 標題路徑（如有）
    full_content: str           # 最終要索引的完整內容
    metadata: Dict


# ─────────────────────────────────────────────
# 方法一：標題路徑前置
# ─────────────────────────────────────────────

def extract_header_path(markdown_text: str) -> List[Tuple[str, str]]:
    """
    解析 Markdown，回傳每段文字及其所屬的標題路徑。
    
    Returns:
        List of (header_path, content) tuples
        例如：("第一章 > 1.1 安裝 > 需求", "Python 3.9 以上...")
    """
    # 使用 LangChain 的 MarkdownHeaderTextSplitter 保留標題層級
    headers_to_split_on = [
        ("#",   "H1"),
        ("##",  "H2"),
        ("###", "H3"),
        ("####","H4"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 保留標題文字在 chunk 中
    )
    docs = splitter.split_text(markdown_text)
    
    results = []
    for doc in docs:
        # 從 metadata 組出 breadcrumb 路徑
        path_parts = []
        for level in ["H1", "H2", "H3", "H4"]:
            if level in doc.metadata:
                path_parts.append(doc.metadata[level])
        
        header_path = " > ".join(path_parts) if path_parts else ""
        results.append((header_path, doc.page_content))
    
    return results


def prepend_header_path(chunks: List[Tuple[str, str]]) -> List[ContextualChunk]:
    """
    將標題路徑前置到每個 chunk，生成 ContextualChunk 列表。
    """
    contextual_chunks = []
    for header_path, content in chunks:
        if header_path:
            full_content = f"{header_path}\n\n{content}"
            context = f"位於章節：{header_path}"
        else:
            full_content = content
            context = ""
        
        contextual_chunks.append(ContextualChunk(
            original_content=content,
            context=context,
            header_path=header_path,
            full_content=full_content,
            metadata={"header_path": header_path}
        ))
    
    return contextual_chunks


# ─────────────────────────────────────────────
# 方法二：Anthropic Contextual Retrieval
# ─────────────────────────────────────────────

# Anthropic 官方建議的 prompt 格式
CONTEXT_PROMPT = """\
<document>
{doc_content}
</document>

以下是需要加入上下文的文件片段：
<chunk>
{chunk_content}
</chunk>

請簡短說明這個片段在整篇文件中的位置與作用，以便在脫離原始文件時仍能被正確理解。
請直接回應說明（100 tokens 以內），不要加入任何前言或解釋。\
"""


def generate_chunk_context(
    client: anthropic.Anthropic,
    doc_content: str,
    chunk_content: str,
    model: str = "claude-3-5-haiku-20241022",
    use_caching: bool = True,
) -> str:
    """
    使用 Claude 為單個 chunk 生成上下文說明。
    
    use_caching=True 時啟用 Anthropic Prompt Caching，
    讓同一份文件的多個 chunk 共享 cache，大幅降低 token 成本。
    """
    prompt = CONTEXT_PROMPT.format(
        doc_content=doc_content,
        chunk_content=chunk_content,
    )
    
    if use_caching:
        # 使用 cache_control 標記文件部分為可快取內容
        # 同一份文件的所有 chunk 只需付一次完整 input token 費用
        response = client.beta.messages.create(
            model=model,
            max_tokens=200,
            betas=["prompt-caching-2024-07-31"],
            system=[{
                "type": "text",
                "text": "你是一個技術文件分析助手，專門為 RAG 系統生成精確的 chunk 上下文說明。",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # 文件內容標記為可快取（同文件多個 chunk 共用）
                        "text": f"<document>\n{doc_content}\n</document>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"以下是需要加入上下文的文件片段：\n"
                            f"<chunk>\n{chunk_content}\n</chunk>\n\n"
                            "請簡短說明這個片段在整篇文件中的位置與作用，"
                            "以便在脫離原始文件時仍能被正確理解。"
                            "請直接回應說明（100 tokens 以內），不要加入任何前言。"
                        ),
                    },
                ],
            }],
        )
    else:
        # 不使用 caching 的簡單版本（適合單次測試）
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
    
    return response.content[0].text.strip()


def batch_generate_contexts(
    client: anthropic.Anthropic,
    doc_content: str,
    chunks: List[str],
    model: str = "claude-3-5-haiku-20241022",
) -> List[ContextualChunk]:
    """
    批次為文件的所有 chunk 生成上下文，並回傳 ContextualChunk 列表。
    啟用 Prompt Caching 以節省成本。
    """
    contextual_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"處理 chunk {i+1}/{len(chunks)}...")
        
        context = generate_chunk_context(
            client=client,
            doc_content=doc_content,
            chunk_content=chunk,
            model=model,
            use_caching=True,  # 同一份文件共享 cache
        )
        
        full_content = f"{context}\n\n{chunk}"
        
        contextual_chunks.append(ContextualChunk(
            original_content=chunk,
            context=context,
            header_path="",
            full_content=full_content,
            metadata={"context": context, "chunk_index": i},
        ))
    
    return contextual_chunks


# ─────────────────────────────────────────────
# 整合：兩種方法組合使用
# ─────────────────────────────────────────────

def process_document_combined(
    doc_content: str,
    anthropic_api_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[ContextualChunk]:
    """
    組合方法：先用標題路徑前置，再用 LLM 補充語意說明。
    適合結構化文件（技術手冊、API 文件等）。
    """
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    # Step 1：用 MarkdownHeaderTextSplitter 按標題切割
    header_chunks = extract_header_path(doc_content)
    
    # Step 2：對過長的 chunk 再做二次切割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    final_chunks = []
    for header_path, content in header_chunks:
        if len(content) > chunk_size:
            # 長段落再切割，每個子 chunk 繼承相同的 header_path
            sub_chunks = text_splitter.split_text(content)
            for sub in sub_chunks:
                final_chunks.append((header_path, sub))
        else:
            final_chunks.append((header_path, content))
    
    # Step 3：為每個 chunk 生成 LLM context 說明
    result = []
    for header_path, chunk_content in final_chunks:
        context = generate_chunk_context(
            client=client,
            doc_content=doc_content,
            chunk_content=chunk_content,
        )
        
        # 組合：標題路徑 + LLM 生成的說明 + 原始內容
        if header_path:
            full_content = f"章節：{header_path}\n{context}\n\n{chunk_content}"
        else:
            full_content = f"{context}\n\n{chunk_content}"
        
        result.append(ContextualChunk(
            original_content=chunk_content,
            context=context,
            header_path=header_path,
            full_content=full_content,
            metadata={
                "header_path": header_path,
                "context": context,
            },
        ))
    
    return result


# ─────────────────────────────────────────────
# 使用範例
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample_doc = """
# 系統設定指南

## 第一章：安裝

### 1.1 系統需求

Python 3.9 以上版本，建議使用 3.11。

### 1.2 安裝步驟

使用 pip 安裝：`pip install myapp`

## 第二章：設定

### 2.1 基本設定

編輯 `config.yml` 填入必要參數。

### 2.2 認證設定

#### JWT Token

預設值為 `true`，可通過 `config.yml` 修改。
"""

    print("=== 方法一：標題路徑前置 ===")
    header_chunks = extract_header_path(sample_doc)
    contextual = prepend_header_path(header_chunks)
    for c in contextual:
        print(f"\n[Header Path] {c.header_path}")
        print(f"[Full Content]\n{c.full_content[:200]}")
        print("---")

    # 方法二需要 API key，僅示意
    # import os
    # chunks = process_document_combined(
    #     doc_content=sample_doc,
    #     anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    # )
```

---

## 優缺點分析

| 面向 | 標題路徑前置 | Anthropic Contextual Retrieval |
|------|-------------|-------------------------------|
| **Retrieval 改善幅度** | 中等（依文件結構而定） | 高（失敗率降低 49%） |
| **實作複雜度** | 低（純文字處理） | 中（需整合 LLM API） |
| **額外成本** | 無 | 每個 chunk 一次 LLM 呼叫 |
| **成本優化手段** | 不需要 | Prompt Caching（可降低 60-80% token 成本） |
| **適用文件類型** | 結構化（Markdown、HTML） | 任意類型 |
| **產出可控性** | 高（格式固定） | 中（依賴 LLM 輸出品質） |
| **延遲影響** | 無（索引期間無額外延遲） | 索引期間延遲增加，查詢無影響 |
| **離線可用性** | 是 | 否（需 API 連線） |
| **搭配 BM25** | 有效 | 效果更顯著（失敗率可降至 1.9%） |

---

## 適用場景

**強烈建議使用的情況：**

- **技術文件與 API 手冊**：章節之間有明確的從屬關係，同一個參數名稱（如 `timeout`）可能出現在多個不同功能的設定段落中，沒有標題上下文幾乎無法區分。

- **產品使用手冊**：使用者提問往往針對特定功能（「如何設定通知？」），但文件中的 chunk 可能只包含設定步驟，沒有明確說明這是「通知設定」的步驟。

- **法律與合規文件**：條款編號和所屬章節是法律文件的核心結構，chunk 脫離章節後語意會產生嚴重歧義。

- **學術論文與技術報告**：摘要、方法論、實驗結果各有其位置，正確的章節標記能讓 RAG 在回答「這篇論文的方法是什麼」時準確召回對應段落。

**效益有限的情況：**

- 非結構化文字（新聞稿、部落格文章）——沒有明確的標題層級可利用。
- chunk 本身已足夠自解釋（chunk size 夠大，包含完整的問答對）。

---

## 與其他方法的比較

| 方法 | 解決問題 | 索引時成本 | 查詢時成本 | 主要限制 |
|------|---------|-----------|-----------|---------|
| **Contextual Headers** | chunk 失去結構資訊 | 低～中 | 無額外成本 | 依賴文件結構 |
| **Parent-Child Chunks** | chunk 粒度與上下文的 trade-off | 低 | 略高（需取回 parent） | 需要二層索引 |
| **HyDE**（假設文件嵌入） | query 與 chunk 語意不匹配 | 無 | 每次查詢一次 LLM 呼叫 | 查詢延遲增加 |
| **Sentence Window** | 精準匹配但上下文不足 | 低 | 略高（需取回鄰句） | 視窗大小難以調整 |
| **Summary Indexing** | 長文件摘要索引 | 高（全文摘要） | 無額外成本 | 細節可能被摘要遺漏 |

**Contextual Headers RAG 的定位**：它在索引階段補強資訊，不增加查詢延遲，是對現有 RAG pipeline 侵入性最小的改進方式之一。它與 Parent-Child Chunks 是互補關係，可以同時使用；與 HyDE 是不同層次的優化（一個從 chunk 側補強，一個從 query 側補強）。

---

## 小結

Contextual Headers RAG 的核心操作只有一個：**在 chunk 進入索引之前，把「它從哪裡來」的資訊補回去**。

兩種實作路徑：

1. **標題路徑前置**：零 API 成本，適合有清晰標題層級的 Markdown/HTML 文件，實作不超過 30 行 Python。
2. **Anthropic Contextual Retrieval**：LLM 生成語意說明，適用任何文件類型，搭配 Prompt Caching 成本可控，Anthropic 實測 retrieval 失敗率降低 49%。

**立刻可以採取的行動：**

- 如果你的文件是 Markdown，現在就可以把 `MarkdownHeaderTextSplitter` 的輸出 metadata 前置到 chunk——不需要任何 API 呼叫。
- 如果你有預算，用 `claude-3-5-haiku` 搭配 Prompt Caching 跑一次 Contextual Retrieval，在索引建立期間多花的成本，會在查詢品質上持續回收。
- 評估改善幅度時，不要只看 NDCG，要直接量測「retrieval 失敗率」（問題明確但 chunk 沒有被召回的比例）——這才是 Contextual Headers 真正在解決的指標。
