<!-- ⚠️ 此檔案由程式自動產生，請勿直接修改。 如需更新內容，請修改來源 JSON 後重新執行 to_preview_md.py。 -->

FID 作為圖像生成評估的主流標準，長期存在幾個被廣泛認知的結構性問題

Inception-v3 是在 ImageNet 上訓練的分類網路，其特徵空間並非專為語意對齊設計

FID 是有偏估計量，樣本數量不足時會系統性地高估生成模型的性能

對 Gaussian 分布的假設在實際圖像特徵分布中很難成立，導致 FID 數值可能失真

CLIP 模型在大規模圖文對上訓練，其視覺特徵具有更豐富的語意對齊能力

CMMD 提出以 CLIP 特徵替代 Inception-v3，並用無偏的 MMD 統計量替代 Fréchet 距離 👇

----
CMMD 使用 CLIP 的 ViT-L/14 視覺編碼器提取生成圖片和真實圖片的特徵向量

最大平均差異（MMD）採用多項式核函數計算兩組特徵分布之間的統計距離

MMD 估計量是無偏的：CMMD = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

其中 k 為多項式核函數，x 代表真實圖片特徵，y 代表生成圖片特徵

CMMD 不需要對特徵分布做任何 Gaussian 假設，適用範圍比 FID 更廣

計算過程不涉及協方差矩陣的平方根估計，避免了 FID 計算中的數值不穩定問題

----
CLIP ViT-L/14 在 400M 圖文對上的對比預訓練使其特徵具有強語意對齊能力

相比 Inception-v3 的 2048 維特徵，CLIP 的特徵維度（1024 維）更集中於語意相關信息

多項式核函數的超參數設定（degree=3）透過在多個評估資料集上的消融實驗確定

研究者測試了不同 CLIP 版本（ViT-B/16、ViT-L/14、ViT-H/14）對 CMMD 分數的影響

開源實現（Google Research GitHub）提供了高效的批量計算代碼，支援 GPU 加速

CMMD 的計算複雜度與 FID 相似，但在小樣本設定下數值穩定性顯著優於 FID

----
在多個圖像生成基準資料集（FFHQ、ImageNet、COCO）上，CMMD 與人類評分的相關性高於 FID

在小樣本設定（1000 張圖片）下，CMMD 的估計偏差顯著小於 FID，適合快速原型評估

對相同一組生成圖片使用不同隨機子集重複計算，CMMD 的變異係數低於 FID

研究者展示了 FID 和 CMMD 在部分情形下給出截然相反的排名，CMMD 更符合人類直覺

在文字條件生成任務（T2I）中，CMMD 的 CLIP 特徵能直接捕捉文字-圖像對齊信息

CMMD 的排名穩定性在不同樣本數（1K 到 50K）下均優於 FID，驗證了無偏估計的實用價值

----
相比 FID，CMMD 在特徵選擇（CLIP vs Inception-v3）和統計框架（MMD vs Fréchet）兩個維度均有改進

相比 KID，CMMD 的核心差異在於使用 CLIP 特徵而非 Inception-v3 特徵，使其更適合 T2I 評估

CMMD 不同於 FID 和 KID 的地方在於其特徵提取器本身就是多模態對齊的，無需額外的文字信息

與 CLIPScore 的逐樣本評估不同，CMMD 仍然是分布層面的指標，提供整體生成品質的統計視角

CMMD 的計算需要 CLIP 環境，而 FID 的計算只需要 PyTorch 和 torchvision，部署成本略高

CMMD 目前在 T2I 評估領域的採用率正在增長，但完全取代 FID 仍需時間，社群慣性依然存在

----
CMMD 由 Google Research 的 Sadeep Jayasumana 等人提出，arxiv：https://arxiv.org/abs/2401.09603

論文以「Rethinking FID」為標題，對圖像生成評估的基本假設進行了系統性的重新審視

CMMD 的開源程式碼已在 GitHub 公開，並被整合到多個圖像生成評估工具庫

研究界對 FID 替代指標的需求在 CMMD 之前就已累積，CMMD 提供了一個有理論依據的解決方案

CMMD 的提出推動了圖像生成評估指標從「能用就好」到「統計嚴謹」的標準提升

後續多篇圖像生成論文開始同時報告 FID 和 CMMD，標誌著評估多元化趨勢的開始

----