# Scratch Transformer Decoder (PyTorch)

PyTorchによる小さなTransformerデコーダのスクラッチ実装です。  
日本語ニュースコーパス（livedoorニュース）を用いて、因果言語モデル（Causal LM）を学習し、文章生成まで行います。

- モデル: Transformer Decoder（GPT風）
- フレームワーク: PyTorch
- データセット: livedoorニュースコーパス
- 環境: Docker / Docker Compose（GPU対応）

## ディレクトリ構成（予定）

```
scratch-transformer-decoder-pytorch/
├── docker/
│ ├── Dockerfile
│ └── requirements.txt
├── docker-compose.yml
├── src/
│ ├── models/
│ │ ├── init.py
│ │ ├── transformer_decoder.py # Transformerデコーダ本体
│ │ └── layers.py # Attention, LayerNorm, FFNなど
│ ├── data/
│ │ ├── init.py
│ │ ├── dataset.py # Datasetクラス
│ │ └── tokenizer.py # トークナイザ
│ ├── train.py # 学習スクリプト
│ ├── generate.py # 生成スクリプト
│ └── utils.py # 補助関数
├── configs/
│ └── model_config.yaml # モデル設定
├── data/ # ニュースコーパス格納先
├── outputs/ # 学習済みモデル・ログ・生成例
└── README.md
```

## 環境構築

### 1. リポジトリのクローン

```bash
git clone https://github.com/avoseven/scratch-transformer-decoder-pytorch.git
cd scratch-transformer-decoder-pytorch
```

### 2. Docker Composeでの環境起動
```bash
docker-compose up -d
docker-compose exec app bash
```
（GPU利用時は docker-compose.yml のGPU設定を有効にしてください）

## 今後の進め方（TODO）

- [ ] ニュースコーパスのダウンロード・展開
- [ ] トークナイザの準備（BPE or 既存vocab）
- [ ] Transformerデコーダのスクラッチ実装
- [ ] Dataset／DataLoaderの実装
- [ ] 学習スクリプト（train.py）の実装
- [ ] 生成スクリプト（generate.py）の実装
- [ ] READMEへの技術解説・生成例の追加
