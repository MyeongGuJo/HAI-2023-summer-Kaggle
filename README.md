# HAI-2023-summer-Kaggle
HAI 2023 여름 방학 과제 (Kaggle)
https://www.kaggle.com/competitions/hai2023summer

baseline 코드에서 모델, lr만 변경

```python
from transformers import BertTokenizer, BertForSequenceClassification # 추가
```
```python
args = easydict.EasyDict({
  "train_path" : "./data/train.csv",
  "valid_path" : "./data/valid.csv",
  "device" : 'cpu',
  "mode" : "train",
  "batch" : 128,
  "maxlen" : 128,
  "lr" : 35e-6, # 변경
  "eps" : 1e-8,
  "epochs" : 1,
  "model_ckpt" : "kykim/bert-kor-base", # 변경
})
```
```python
# load model and tokenizer
# CHECKPOINT_NAME = 'kykim/bert-kor-base'
model = BertForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=3) # 변경
model.to(args.device)
tokenizer = BertTokenizer.from_pretrained(args.model_ckpt) # 변경
```

사용 모델: https://huggingface.co/kykim/bert-kor-base


---
8.19 수정

모델과 lr을 변경하니 점수가 향상됨. (0.94408 -> 0.97444)
모델은 내가 찾은 건 아니고 지명이가 올린 걸 돌려본 것...

```python
from transformers import ElectraTokenizer, ElectraForSequenceClassification # 추가
```
```python
args = easydict.EasyDict({
  "train_path" : "./data/train.csv",
  "valid_path" : "./data/valid.csv",
  "device" : 'cpu',
  "mode" : "train",
  "batch" : 128,
  "maxlen" : 128,
  "lr" : 4e-5, # 변경
  "eps" : 1e-8,
  "epochs" : 1,
  "model_ckpt" : "circulus/koelectra-dialect-v1", # 변경
})
```
```python
# load model and tokenizer
# CHECKPOINT_NAME = "circulus/koelectra-dialect-v1"
model = ElectraForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=3) # 변경
model.to(args.device)
tokenizer = ElectraTokenizer.from_pretrained(args.model_ckpt) # 변경
```

사용 모델: https://huggingface.co/circulus/koelectra-dialect-v1
