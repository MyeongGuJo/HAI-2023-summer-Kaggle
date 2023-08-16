# HAI-2023-summer-Kaggle
HAI 2023 여름 방학 과제 (Kaggle)
https://www.kaggle.com/competitions/hai2023summer

baseline 코드에서 모델, lr만 변경
```python
from transformers import BertTokenizer, BertForSequenceClassification
```
```
# load model and tokenizer
# CHECKPOINT_NAME = 'kykim/bert-kor-base'
model = BertForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=3)
model.to(args.device)
tokenizer = BertTokenizer.from_pretrained(args.model_ckpt)
```

lr = 35e-6으로 설정
