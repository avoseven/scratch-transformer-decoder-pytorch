import os
import glob
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def clean_text(text) -> str:
# ... (既存のclean_text関数)
    """
    テキストのクリーニング処理
    - 改行やタブの除去
    - URLの除去
    - 重複する空白の整理
    """
    # URLの除去
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    # 改行、タブ、特殊文字の置換
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    # 重複する空白を1つに
    text = re.sub(r'\s+', ' ', text)
    # 前後の空白削除
    text = text.strip()
    return text

def load_news_corpus(data_dir: str='data/text') -> list[str]:
    """
    Livedoorニュースコーパスを読み込む
    各カテゴリのディレクトリからテキストファイルを取得し、タイトルと本文を抽出する
    """
    all_texts = []
    
    # カテゴリディレクトリの取得（LICENSE.txtなどを除外するためディレクトリのみ取得）
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Loading corpus from {data_dir}...")
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        file_paths = glob.glob(os.path.join(category_dir, f"{category}-*.txt"))
        
        for path in tqdm(file_paths, desc=f"Category: {category}"):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 0: URL, 1: タイムスタンプ, 2: タイトル, 3以降: 本文
                if len(lines) > 2:
                    title = lines[2].strip()
                    body = "".join(lines[3:]).strip()
                    full_text = f"{title} {body}"
                    cleaned_text = clean_text(full_text)
                    if cleaned_text:
                        all_texts.append(cleaned_text)
                        
    print(f"Total samples loaded: {len(all_texts)}")
    return all_texts

def get_train_val_datasets(texts, test_size=0.1, random_state=42):
    """
    データを学習用と検証用に分割する
    """
    train_texts, val_texts = train_test_split(
        texts, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
    return train_texts, val_texts

class NewsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Decoderモデルの学習用に入力(x)とターゲット(y)を作るため、max_length + 1 でエンコード
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length + 1,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # 入力x: 最初から最後の一つ前まで
        # ターゲットy: 二つ目から最後（一つ右にシフトしたもの）まで
        x = input_ids[:-1]
        y = input_ids[1:].clone()
        
        # ターゲットのパディング部分を -100 に置き換えて Loss 計算から除外する
        # シフトした後の attention_mask も同様に扱う
        target_mask = attention_mask[1:]
        y[target_mask == 0] = -100
        
        # 入力側の attention_mask もシフトに合わせて1つ分削る
        x_mask = attention_mask[:-1]
        
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": x_mask
        }

if __name__ == "__main__":
    # 動作確認用
    from transformers import AutoTokenizer
    
    # ダミーのテキストデータ
    texts = [
        "こんにちは、これはテストです。",
        "PyTorchでTransformerをスクラッチから実装しています。",
        "日本語のサブワードトークナイザーを使用します。"
    ]
    
    # 日本語対応のトークナイザー（軽量なものとして bert-base-japanese を例に使用）
    # ※ 実行環境に依存するため、エラー時は適宜変更してください
    try:
        tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        dataset = NewsDataset(texts, tokenizer, max_length=16)
        
        print(f"Dataset length: {len(dataset)}")
        sample = dataset[0]
        print("\nSample keys:", sample.keys())
        print("Input IDs shape:", sample["input_ids"].shape)
        print("Labels shape:", sample["labels"].shape)
        print("Attention Mask shape:", sample["attention_mask"].shape)
        print("\nInput IDs:", sample["input_ids"])
        print("Labels:", sample["labels"])
    except Exception as e:
        print(f"Tokenizer loading failed (this is expected if libraries are not installed yet): {e}")
