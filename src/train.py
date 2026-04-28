import os
import time
import math
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.transformer import Transformer, TransformerConfig
from data.dataset import NewsDataset, load_news_corpus
from data.tokenizer import JapaneseTokenizer

import argparse

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, iteration, loss, config, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

def get_lr(it, train_config):
    # 1) warmup期間
    if it < train_config['warmup_iters']:
        return train_config['learning_rate'] * it / train_config['warmup_iters']
    # 2) it > lr_decay_iters なら最小値
    if it > train_config['lr_decay_iters']:
        return train_config['min_lr']
    # 3) cosine decay
    decay_ratio = (it - train_config['warmup_iters']) / (train_config['lr_decay_iters'] - train_config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config['min_lr'] + coeff * (train_config['learning_rate'] - train_config['min_lr'])

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, eval_iters):
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        iter_loader = iter(loader)
        for k in range(eval_iters):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                batch = next(iter_loader)
            
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            mask = batch['attention_mask'].to(device)
            _, loss = model(x, targets=y, attention_mask=mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='latestチェックポイントから学習を再開する')
    args = parser.parse_args()

    # 設定のロード
    config = load_config()
    m_cfg = config['model']
    t_cfg = config['train']
    p_cfg = config['paths']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
        device_type = 'mps'
    else:
        device_type = 'cpu'
    
    os.makedirs(p_cfg['checkpoint_dir'], exist_ok=True)

    # 1. トークナイザーとデータのロード
    tokenizer = JapaneseTokenizer(p_cfg['tokenizer_model'])
    print("Loading Livedoor news corpus...")
    all_texts = load_news_corpus(p_cfg['data_dir'])
    train_texts, val_texts = train_test_split(all_texts, test_size=0.1, random_state=42)
    
    train_dataset = NewsDataset(train_texts, tokenizer, max_length=m_cfg['block_size'])
    val_dataset = NewsDataset(val_texts, tokenizer, max_length=m_cfg['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=t_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=t_cfg['batch_size'], shuffle=False)

    # 2. モデルの初期化
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=m_cfg['block_size'],
        n_layer=m_cfg['n_layer'],
        n_head=m_cfg['n_head'],
        n_embd=m_cfg['n_embd'],
        dropout=m_cfg['dropout'],
        bias=m_cfg['bias']
    )
    model = Transformer(model_config).to(device)

    # 3. オプティマイザ
    optimizer = model.configure_optimizers(
        t_cfg['weight_decay'], t_cfg['learning_rate'], 
        (t_cfg['beta1'], t_cfg['beta2']), device_type
    )

    # --- 再開（Resume）処理 ---
    start_iter = 0
    if args.resume:
        # 拡張子が .pt のファイルをすべて取得
        checkpoint_files = [f for f in os.listdir(p_cfg['checkpoint_dir']) if f.endswith(".pt")]
        if checkpoint_files:
            # ファイルの更新日時（mtime）が最新のものを取得
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(p_cfg['checkpoint_dir'], x)), reverse=True)
            latest_ckpt = os.path.join(p_cfg['checkpoint_dir'], checkpoint_files[0])
            
            print(f"Resuming from the latest checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            print("No checkpoints found. Starting from scratch.")
    # -----------------------

    # 4. 学習ループ
    iter_loader = iter(train_loader)
    print(f"Starting training on {device} from iter {start_iter}...")
    start_time = time.time()

    try:
        for i in range(start_iter, t_cfg['max_iters']):
            # 学習率の更新
            lr = get_lr(i, t_cfg)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 定期的なバリデーション
            if i % t_cfg['eval_interval'] == 0:
                losses = estimate_loss(model, train_loader, val_loader, device, t_cfg['eval_iters'])
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")

            # データの取得
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            mask = batch['attention_mask'].to(device)

            # Forward / Backward
            logits, loss = model(x, targets=y, attention_mask=mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if t_cfg['grad_clip'] != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg['grad_clip'])
            optimizer.step()

            if i % 10 == 0:
                print(f"iter {i}: loss {loss.item():.4f}")

            # 保存
            if i > 0 and i % t_cfg['save_interval'] == 0:
                save_checkpoint(model, optimizer, i, loss.item(), model_config, p_cfg['checkpoint_dir'], f"ckpt_iter_{i}.pt")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        save_checkpoint(model, optimizer, i, loss.item(), model_config, p_cfg['checkpoint_dir'], "ckpt_interrupted.pt")
        print("Done.")

    print(f"Training finished! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
