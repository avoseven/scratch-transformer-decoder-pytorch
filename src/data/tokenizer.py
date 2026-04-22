from transformers import AutoTokenizer
import os

class JapaneseTokenizer:
    """
    HuggingFaceのAutoTokenizerを使用した日本語用トークナイザークラス。
    """
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-v3"):
        """
        Args:
            model_name: 使用する学習済みモデル名、またはローカルディレクトリのパス。
        """
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Decoderモデルの学習にはパディングトークンが必要。
        # 設定されていない場合は eos_token を代用する。
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # どちらもない場合は新しく追加（通常はどちらかがある）
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
    def encode(self, text: str, max_length: int = None, padding: bool = False, truncation: bool = True):
        """テキストをトークンIDに変換する"""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """トークンIDをテキストに変換する"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        """語彙サイズを返す"""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id or self.tokenizer.cls_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

    def save(self, save_dir: str):
        """トークナイザーの設定を保存する"""
        self.tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    # 動作確認
    tokenizer = JapaneseTokenizer()
    text = "吾輩は猫である。名前はまだ無い。"
    
    encoded = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Encoded IDs: {encoded['input_ids']}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"Decoded text: {decoded}")
