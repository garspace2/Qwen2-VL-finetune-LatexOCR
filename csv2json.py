import pandas as pd
import json

csv_path = './latex_ocr_train.csv'
json_path = './latex_ocr_train.json'

df = pd.read_csv(csv_path)
# Create conversation format
conversations = []
prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"

# Add image conversations
for i in range(len(df)):
    conversations.append({
        "id": f"identity_{i+1}",
        "conversations": [
            {
                "from": "user",
                "value": f"{prompt} <|vision_start|>{df.iloc[i]['image_path']}<|vision_end|>"
            },
            {
                "from": "assistant", 
                "value": str(df.iloc[i]['text'])
            }
        ]
    })

print(conversations)
# Save to JSON
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)
