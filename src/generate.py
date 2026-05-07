import os
import yaml
import torch
import torch.nn.functional as F
from models.transformer import Transformer, TransformerConfig
from data.tokenizer import JapaneseTokenizer
import argparse

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None, device='cpu'):
    """
    モデルを使用してテキストを生成する。
    
    Args:
        model: Transformerモデル
        tokenizer: JapaneseTokenizerインスタンス
        prompt: 入力テキスト
        max_new_tokens: 生成する最大トークン数
        temperature: サンプリングの温度。高いほど多様、低いほど決定的。
        top_k: Top-KサンプリングのK。上位K個のトークンからのみサンプリングする。
        device: 使用デバイス
    """
    model.eval()
    
    # プロンプトをトークナイズ
    encoded = tokenizer.encode(prompt)
    idx = encoded['input_ids'].to(device) # (1, seq_len)

    # プロンプトの末尾がEOS（通常2）の場合、それを削除してから生成を開始する
    if idx[0, -1] == tokenizer.eos_token_id:
        idx = idx[:, :-1]
    
    # 生成ループ
    for _ in range(max_new_tokens):
        # コンテキスト窓（block_size）を超えないようにクロップ
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        # モデルのフォワード (targets=Noneなので最後のトークンの予測結果のみ返ってくる)
        logits, _ = model(idx_cond) # logits: (1, 1, vocab_size)
        
        # 最後のタイムステップのロジットを取り出し、温度を適用
        logits = logits[:, -1, :] / temperature
        
        # Top-K フィルタリング
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 確率分布に変換
        probs = F.softmax(logits, dim=-1)
        
        # サンプリング
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # シーケンスに追加
        idx = torch.cat((idx, idx_next), dim=1)
        
        # EOSトークンが出たら終了
        if idx_next.item() == tokenizer.eos_token_id:
            break
            
    # 特殊トークンを除いてデコード
    return tokenizer.decode(idx[0])

def main():
    parser = argparse.ArgumentParser(description='Transformer Decoder テキスト生成')
    parser.add_argument('--prompt', type=str, default="今日", help='生成の起点となるテキスト')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='追加で生成する最大トークン数')
    parser.add_argument('--temperature', type=float, default=0.8, help='サンプリングの温度 (0.0~, デフォルト: 0.8)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-KサンプリングのK (デフォルト: 50)')
    parser.add_argument('--checkpoint', type=str, default=None, help='使用するチェックポイントのパス (未指定なら最新を使用)')
    args = parser.parse_args()

    # 設定のロード
    config = load_config()
    
    # デバイスの設定
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # トークナイザーのロード
    tokenizer = JapaneseTokenizer(config['paths']['tokenizer_model'])
    
    # チェックポイントの決定
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_dir = config['paths']['checkpoint_dir']
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not ckpt_files:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        # ファイルの更新日時が最新のものを取得
        ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_files[0])
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # モデルの初期化と重みのロード
    # TransformerConfigの代わりにcheckpointからconfigを復元
    model_config = checkpoint['config']
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 30)
    
    # テキスト生成の実行
    generated_text = generate(
        model, 
        tokenizer, 
        args.prompt, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k,
        device=device
    )
    
    print(generated_text)
    print("-" * 30)

if __name__ == "__main__":
    main()
