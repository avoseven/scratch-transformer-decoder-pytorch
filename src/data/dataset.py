import os
import glob
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def clean_text(text) -> str:
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

if __name__ == "__main__":
    # 動作確認用
    texts = load_news_corpus()
    train, val = get_train_val_datasets(texts)
    if train:
        print("\nExample processed text:")
        print(train[0][:200] + "...")
