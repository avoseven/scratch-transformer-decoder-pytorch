import os
import sentencepiece as spm
from dataset import load_news_corpus

def train_sentencepiece_tokenizer(data_dir: str = "data/text", vocab_size: int = 8000, model_prefix: str = "data/tokenizer/news_spm"):
    """
    SentencePieceを使用して独自のトークナイザーを学習する
    """
    # 保存先ディレクトリの作成
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
    # 1. コーパスの読み込みとテキストファイルの作成
    texts = load_news_corpus(data_dir)
    train_text_file = "data/tokenizer/corpus.txt"
    
    print(f"Saving corpus to {train_text_file} for training...")
    with open(train_text_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
            
    # 2. SentencePieceの学習
    # --input: 学習用テキスト
    # --model_prefix: 出力モデル名
    # --vocab_size: 語彙サイズ
    # --character_coverage: 日本語の場合は 0.9995 が推奨
    # --model_type: bpe (Byte Pair Encoding)
    # --pad_id, --bos_id, --eos_id, --unk_id: 特殊トークンのID指定
    print(f"Training SentencePiece (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=train_text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="[PAD]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        unk_piece="[UNK]"
    )
    
    print(f"Training finished! Model saved at {model_prefix}.model")
    # 一時ファイルの削除
    if os.path.exists(train_text_file):
        os.remove(train_text_file)

if __name__ == "__main__":
    train_sentencepiece_tokenizer()
