# KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs

KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs
免重算且情境無關的 LLM KV 快取機制
LLM 推論速度很大程度靠 KV 快取撐著
但傳統 KV 快取是情境相依的
同一份文件換個上下文就得重算 KV
現有方案雖然減少部份重算，仍然有不能忽略的浮點運算和首個 token 延遲
KV Packet 想把重算這件事乾脆拿掉
👇

---

概念很直接
把快取過的文件當成不可變的 packet
每個 packet 外面套上一層輕量可訓練的 soft token adapter
adapter 透過自監督蒸餾訓練
目的是在不同上下文間補平 KV 分佈的差異
這樣一來重算就變成不必要的動作

---

這個設計的重點是可插拔
packet 本身不動
只要換個 adapter 就能改變它適用的情境
部署端直接接上就能用，不需要再進模型算一次
這像是把文件包成標準套件，整個服務端的架構也變得更乾淨

---

在 Llama-3.1 和 Qwen2.5 上的實驗結果明顯
KV Packet 的浮點運算幾乎歸零
首個 token 時間比重算式基線還短
F1 分數維持在完整重算基線的水準
這三個指標同時拿到，代表方法在品質與效率間沒有明顯取捨

---

這類設計對企業 RAG 場景特別重要
同一份政策文件與同一本說明書被反覆檢索
如果每次都重算 KV，GPU 會被吃掉很多算力
把文件當 packet 包好存起來
每次查詢只付 adapter 的代價
這對吞吐量來說是質變

---

KV Packet 反映了一個很實用的方向
與其優化原本的重算流程，不如重新思考快取本身能否更抽象
當你把文件視為獨立單位
LLM 推論基建的空間就被重新打開了
這是工程思維帶來的真實收益
