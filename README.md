# Make Data Count + LLaMA 3

本專案專為 Kaggle 的 Make Data Count 比賽設計，使用 LLaMA 3 8B Instruct 為核心模型，整合 Self-Evolving Learning (SEL) 四大模組進行文獻資料引用分類。

📌 開發目標：
- 輸入格式：PDF / XML
- 輸出格式：submission.csv
- Leaderboard > 0.910 分數

📁 專案結構對應：
- L1–L3：預處理與基礎推理
- L4–L5：精煉與再訓練（Self-Refinement + Meta-Cog）
- L6–L7：記憶檢索與結果產出驗證

```mermaid
graph TD
A[PDF/XML輸入] --> B[ID擷取+NER]
B --> C[Sliding Window 建 Context]
C --> D[LLM 推理（Primary/Secondary）]
D --> E{分類置信度低？}
E -->|是| F[Self-Refinement 啟動]
F --> G[錯誤樣本紀錄]
G --> H[CIT prompt pair 產生]
H --> I[LoRA 微調 LLaMA 3]
I --> J[回到 D 進行改進推理]
E -->|否| K[寫入 submission.csv]
```
