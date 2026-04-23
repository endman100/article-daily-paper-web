<!-- ⚠️ 此檔案由程式自動產生，請勿直接修改。 如需更新內容，請修改來源 JSON 後重新執行 to_preview_md.py。 -->

現有的 T2I 評估指標（CLIPScore、FID）在捕捉人類實際偏好方面存在系統性偏差

訓練符合人類偏好的評估模型需要大規模、真實的人類偏好標注資料，而非合成資料

PickScore 基於 Pick-a-Pic 平台收集的真實用戶互動偏好資料訓練，反映了真實使用行為

Pick-a-Pic 資料集包含超過 50 萬組由真實用戶在使用 T2I 工具時做出的成對偏好標注

這些資料來自非實驗室環境的自然使用行為，有更高的生態效度（ecological validity）

PickScore 在預測人類偏好方面的準確率顯著優於基於合成或小規模資料訓練的替代指標 👇

----
PickScore 在 CLIP ViT-H/14 基礎上進行偏好學習 fine-tuning

訓練目標使用對比損失（Contrastive Loss）：對同一 prompt，使人類偏好的圖片得分高於非偏好的

Pick-a-Pic 資料集的構建：用戶在 Stable Diffusion XL 的 web 界面中使用 prompt 生成多張圖片並選擇最喜歡的

每個訓練樣本包含：文字 prompt、被選圖片、未被選圖片，形成三元組

PickScore 的最終輸出是給定 prompt-image 對的標量偏好分數

推論時 PickScore 可以直接計算單個圖片的分數，也可以比較多個候選圖片的相對得分

----
Pick-a-Pic 資料集的收集方法：在真實 T2I 應用中嵌入 A/B 測試界面，記錄用戶的自然選擇

資料多樣性：涵蓋藝術、照片、插圖、概念藝術等多種風格，prompt 來自真實用戶的自然語言輸入

數據清洗：過濾掉明顯的隨機點擊（停留時間極短的判斷）和重複用戶的系統性偏見

訓練使用 AdamW 優化器，學習率 warmup 策略，在 A100 GPU 上訓練約 10 萬步

CLIP ViT-H/14 的大規模預訓練使 PickScore 能夠利用豐富的視覺語言語意知識

PickScore 的推論速度與 CLIPScore 相近，可以在評估流程中作為直接替換

----
在 PartiPrompts 和 DrawBench 的人類偏好預測任務上，PickScore 的準確率比 CLIPScore 高約 10%

在 COCO 圖片偏好預測任務上，PickScore 的預測準確率達到 65-70%（隨機基線 50%）

在跨模型比較任務（Stable Diffusion 1.5 vs DALL-E 2）中，PickScore 的排名與人類評估高度一致

Pick-a-Pic 資料集包含來自 50,000+ 不同 prompt 的偏好判斷，確保了訓練資料的多樣性

消融研究顯示，使用真實用戶資料（vs 眾包標注）訓練的 PickScore 在分布外測試上表現更穩健

PickScore 與 HPS v2 的性能比較顯示兩者在不同 prompt 類別上各有優劣

----
相比 CLIPScore，PickScore 對美學偏好的敏感度更高，因為訓練信號包含了人類的美學判斷

相比 ImageReward，PickScore 的訓練資料來自自然使用行為而非眾包標注，更能反映真實偏好

相比 HPS v2，PickScore 使用了更大型的基礎模型（ViT-H vs ViT-L），在複雜場景下泛化更強

PickScore 的主要局限是其訓練資料主要來自 Stable Diffusion 用戶，對其他模型的偏好可能有遷移誤差

在評估高度攝影寫實的圖片時，PickScore 的表現優於藝術或概念設計類的生成圖片

PickScore 與所有基於 CLIP 的指標共享的弱點：對否定語意和複雜組合關係的評估能力有限

----
PickScore 和 Pick-a-Pic 資料集由 Yuval Kirstain 等人發表，arxiv：https://arxiv.org/abs/2305.01569

Pick-a-Pic 提供了一個從真實用戶行為收集 T2I 偏好資料的可重複框架，對後續研究有重要示範作用

PickScore 被廣泛用作 T2I 模型優化的 reward model，特別是在 RLHF 訓練流程中

基於 PickScore 的 reward signal 已被用於 Stable Diffusion 的 RLHF 微調，驗證了其作為訓練信號的有效性

PickScore 的成功推動了多個開放 T2I 偏好資料集的建立（如 HPD v2、ImageReward 資料集擴充）

未來方向包括使用更多元化的用戶群體資料更新訓練集，以減少 Stable Diffusion 用戶偏好的系統性偏差

----