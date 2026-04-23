import torch
import sys
import os

# srcディレクトリをパスに追加してインポート可能にする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import NewsDataset

class MockTokenizer:
    """NewsDatasetのロジックをテストするためのモックトークナイザー"""
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.pad_token = "[PAD]"

    def __call__(self, text, truncation=True, max_length=10, padding="max_length", return_tensors="pt"):
        # テキストを単に文字コードに変換するような簡易的なトークナイズ
        ids = [ord(c) % 100 + 1 for c in text[:max_length]] # 0を避けるため+1
        
        attention_mask = [1] * len(ids)
        
        # パディング処理
        if len(ids) < max_length:
            padding_len = max_length - len(ids)
            ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attention_mask])
        }

def test_news_dataset_logic():
    print("Testing NewsDataset logic...")
    
    texts = ["abc", "hello world"]
    max_length = 5
    pad_id = 0
    tokenizer = MockTokenizer(pad_token_id=pad_id)
    
    dataset = NewsDataset(texts, tokenizer, max_length=max_length)
    
    # 1つ目のサンプル (text="abc", length=3 < max_length=5)
    # dataset.__getitem__ は内部で max_length + 1 (=6) を要求する
    sample = dataset[0]
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample["attention_mask"]
    
    print(f"Input IDs: {input_ids}")
    print(f"Labels:    {labels}")
    print(f"Mask:      {attention_mask}")
    
    # 検証1: 形状の確認 (max_length と一致すべき)
    assert input_ids.shape[0] == max_length
    assert labels.shape[0] == max_length
    assert attention_mask.shape[0] == max_length
    
    # 検証2: シフトの確認 (labels[i] == input_ids[i+1] であるべき、パディング以外)
    # MockTokenizerの内部では max_length+1=6 トークン生成される
    # x = ids[0:5], y = ids[1:6]
    # labelsが正しくシフトされているか（パディングでない場所）
    for i in range(max_length - 1):
        if labels[i] != -100:
            # labels[i] は元の ids[i+1]
            # input_ids[i+1] も元の ids[i+1]
            assert labels[i] == input_ids[i+1], f"Mismatch at index {i}"

    # 検証3: パディングの置換確認
    # text="abc" (3トークン) なので、max_length+1=6 のうち 4,5,6番目がパディング(0)
    # x = [a, b, c, 0, 0]
    # y = [b, c, 0, 0, 0] -> マスク処理で [b, c, -100, -100, -100]
    assert labels[2] == -100, "Padding should be replaced by -100"
    assert labels[3] == -100
    assert labels[4] == -100
    
    print("Logic test passed!")

if __name__ == "__main__":
    test_news_dataset_logic()
