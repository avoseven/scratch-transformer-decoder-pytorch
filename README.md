# Scratch Transformer Decoder (PyTorch)

PyTorchによる小さなTransformerデコーダのスクラッチ実装です。  
日本語ニュースコーパス（livedoorニュース）を用いて、因果言語モデル（Causal LM）を学習し、文章生成まで行います。

- モデル: Transformer Decoder（GPT風、13.8M params）
- フレームワーク: PyTorch
- データセット: livedoorニュースコーパス
- 環境: Docker / Docker Compose

## 使い方（学習実行）

### 1. 環境の起動
```bash
docker compose up -d
```

### 2. 学習の開始（新規）
バックグラウンドで学習を開始し、ログを `outputs/train.log` にリアルタイムで記録します。
```bash
docker compose exec -d app sh -c "python -u src/train.py > outputs/train.log 2>&1"
```

### 3. 学習の再開（Resume）
最新のチェックポイント（`outputs/checkpoints/` 内の最新ファイル）から学習を再開します。
```bash
docker compose exec -d app sh -c "python -u src/train.py --resume >> outputs/train.log 2>&1"
```

### 4. 進捗の確認
```bash
tail -f outputs/train.log
```

### 5. 安全な中断
計算を安全に停止し、その瞬間の状態を `ckpt_interrupted.pt` として保存します。
```bash
docker compose exec app pkill -INT -f "python src/train.py"
```

## ディレクトリ構成

```
scratch-transformer-decoder-pytorch/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── src/
│   ├── models/
│   │   └── transformer.py     # Transformerデコーダ本体
│   ├── data/
│   │   ├── dataset.py         # Datasetクラス
│   │   └── tokenizer.py       # トークナイザ
│   ├── train.py               # 学習スクリプト（Resume/中断機能付き）
│   └── generate.py            # 生成スクリプト
├── configs/
│   └── model_config.yaml      # モデル・学習設定
├── data/                      # コーパス・トークナイザーモデル
├── outputs/                   # チェックポイント・ログ
└── README.md
```

## 進捗状況

- [x] ニュースコーパスの準備
- [x] トークナイザの実装（JapaneseTokenizer）
- [x] Transformerデコーダのスクラッチ実装（Flash Attention対応）
- [x] Dataset／DataLoaderの実装
- [x] 学習スクリプト（train.py）の実装（Resume機能、中断保存機能）
- [ ] 生成スクリプト（generate.py）の調整
- [ ] 大規模学習の実施
- [ ] READMEへの技術解説・生成例の追加
