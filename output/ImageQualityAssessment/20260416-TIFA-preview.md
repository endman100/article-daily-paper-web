<!-- ⚠️ 此檔案由程式自動產生，請勿直接修改。 如需更新內容，請修改來源 JSON 後重新執行 to_preview_md.py。 -->

評估 T2I 生成圖片與輸入文字 prompt 之間的語意符合度是確保模型可用性的關鍵

CLIPScore 等基於嵌入相似度的指標在評估細粒度語意元素時存在粒度不足的問題

「一隻穿著紅色外套的狗在雪地裡玩球」——CLIPScore 無法精確判斷每個語意元素是否出現

問答（QA）框架提供了一種更精確、可解釋的語意符合度評估方法

TIFA（Text-Image Faithfulness via Question Answering）系統化地將 prompt 分解為可驗證的問題

透過逐一驗證每個語意元素的存在，TIFA 提供了細粒度、可解釋的語意符合度評分 👇

----
TIFA 的評估流程包含兩個主要步驟：問題生成和視覺問答

問題生成：使用 GPT-3.5/4 從文字 prompt 自動生成一組是非題，覆蓋所有語意元素

問題類型涵蓋：物件存在（object）、屬性（attribute）、關係（relation）、計數（counting）

視覺問答：使用 MLLM（InstructBLIP、PaLI-X 等）回答每個問題

TIFA 分數為所有問題中回答正確的比例（0-1 範圍）

問題生成策略確保涵蓋 prompt 中所有重要的語意成分，使評估具有完整性

----
TIFA 的問題生成使用少樣本提示（few-shot prompting）確保問題品質和一致性

問題過濾步驟：使用真實圖片（GT image）驗證問題的可回答性，過濾掉 VQA 模型無法正確回答的問題

VQA 模型的選擇顯著影響 TIFA 的準確性：更強大的 MLLM 提升問答準確性

TIFA 基準資料集包含從 4000 個 prompt 生成的 25,000+ 個問題，可作為標準評估套件使用

評估支持多個 VQA 後端的即插即用設計，使 TIFA 能夠受益於 VQA 技術的持續進步

問題級別的分析使研究者能夠識別 T2I 模型在哪類語意元素（計數、關係等）上表現弱

----
在 TIFA 基準上，現代 T2I 模型（DALL-E 3、Stable Diffusion XL）的 TIFA 分數差距揭示了語意符合度的顯著差異

在計數類問題上，所有測試的 T2I 模型均表現顯著低於物件存在類問題

TIFA 分數與人類的語意符合度評分的 Spearman 相關係數約為 0.7-0.8

在空間關係類問題（left/right/above/below）上，T2I 模型的 TIFA 分數通常在 0.4-0.6 之間

使用 GPT-4V 作為 VQA 後端的 TIFA 比 BLIP-2 後端提升了約 10% 的人類一致性

TIFA 的問題類別分析揭示 T2I 模型的規律性弱點，直接指引模型改進的方向

----
相比 CLIPScore，TIFA 在細粒度語意評估上更精確，且提供可解釋的問題級別分析

相比 VISOR，TIFA 更通用（不限於空間關係），但 VISOR 在空間語意評估上更為精確

相比 VQAScore，TIFA 使用預定義問題集，可重現性更高；VQAScore 使用單一問題，更簡潔

TIFA 的主要局限是計算成本：每張圖片需要調用 LLM（問題生成）和 MLLM（問題回答）

問題生成的隨機性可能導致不同 TIFA 評估運行的問題集略有差異，影響可重現性

TIFA 對 prompt 複雜度敏感：簡單 prompt 的問題集太少，複雜 prompt 的問題集過大

----
TIFA 由 Yushi Hu 等人在 EMNLP 2023 發表，arxiv：https://arxiv.org/abs/2303.11897

TIFA 的發表推動了 QA-based T2I 評估框架的廣泛採用，成為語意符合度評估的標準方法之一

TIFA 基準資料集（TIFA v1.0）已被多個後續研究工作用作評估語意符合度的標準測試集

基於 TIFA 框架的改進工作（SoftTIFA、DSG）探索了問題生成和評分機制的精細化

TIFA 的可解釋性優勢（問題級分析）啟發了後續研究對 T2I 評估診斷能力的重視

隨著 MLLM 能力的快速進步，TIFA 的問答後端選擇對評估準確性的影響越來越小

----