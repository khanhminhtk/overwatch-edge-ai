import json
import torch
import cv2

# with open('/home/minhtk/code/overwatch-edge-ai/worktree/ai_forge/ai_forge/data/recognizer/train_data/1/label.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# for key, value in data.items():
#     print(f"{key}: {value}")

image = cv2.imread('/home/minhtk/code/overwatch-edge-ai/worktree/ai_forge/ai_forge/data/recognizer/train_data/1/1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
torch_image = torch.from_numpy(image).float()
print(torch_image.shape)