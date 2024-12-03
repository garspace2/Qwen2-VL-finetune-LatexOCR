import pandas as pd
import json

csv_path = './latex_ocr_train.csv'
train_json_path = './latex_ocr_train.json'
val_json_path = './latex_ocr_val.json'
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

# print(conversations)
# Save to JSON
# Split into train and validation sets
train_conversations = conversations[:-4]
val_conversations = conversations[-4:]

# Save train set
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f, ensure_ascii=False, indent=2)

# Save validation set 
with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)
