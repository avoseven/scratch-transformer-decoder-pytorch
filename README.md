# Scratch Transformer Decoder (PyTorch)

PyTorchによるTransformer DecoderのScratch実装
- CPU環境で学習できる程度の小規模Modelを対象とする
- 日本語ニュースコーパス（livedoorニュース）を用いて、因果言語モデル（Causal LM）を学習し、文章生成まで行う

- モデル: Transformer Decoder（GPT風、13.8M params）
- フレームワーク: PyTorch
- データセット: livedoorニュースコーパス
- 環境: Docker / Docker Compose

下記の流れで現在に至る
1. Transformer Decoderを実装し，0から学習させたが，生成能力は芳しくなかった
2. 公開Modelを導入し，追加学習を実施したが，なおも生成能力は芳しくなかった
3. 現時点での結論としては，公開Modelをそのまま流用するのが最良であるとする

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
docker compose exec app tail -f outputs/train.log
```

### 5. 安全な中断
計算を安全に停止し、その瞬間の状態を `ckpt_interrupted.pt` として保存します。
```bash
docker compose exec app pkill -INT -f "src/train.py"
```

## 使い方 (文章生成)
最新のチェックポイントを使用して文章を生成します．
```bash
docker compose exec app python3 src/generate.py --prompt "私は" --checkpoint outputs/checkpoints/ckpt_final.pt
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
│   │   ├── __init__.py
│   │   └── transformer.py        # Transformerデコーダ本体
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Datasetクラス
│   │   ├── tokenizer.py          # トークナイザ
│   │   └── train_tokenizer.py    # トークナイザの学習
│   ├── utils.py
│   ├── train.py                  # 学習スクリプト（Resume/中断機能付き）
│   └── generate.py               # 生成スクリプト (EOS除去修正済)
├── tests
│   └── test_dataset_logic.py
├── configs/
│   └── model_config.yaml         # モデル・学習設定
├── data/                         # コーパス・トークナイザーモデル
│   ├── text/                     # 学習用Text data
│   └── tokenizer/                # 自前Tokenizer
│   │   ├── news_spm.model
│   │   └── news_spm.vocab
├── outputs/
│   ├── checkpoints/              # Checkpoint
│   └── train.log                 # Log
├── .gitignore
└── README.md
```

## 1. 自前実装
Transformer Decoderを実装し，0から学習させたが，生成能力は芳しくなかった

### 進捗状況
- [x] ニュースコーパスの準備
- [x] Dataset／DataLoaderの実装
- [x] Tokenizerの実装（JapaneseTokenizer）
- [x] Tokenizerの学習
- [x] Transformerデコーダのスクラッチ実装（Flash Attention対応）
- [x] 学習スクリプト（train.py）の実装（Resume機能、中断保存機能）
- [x] 生成スクリプト（generate.py）の実装
- [x] 本番学習の実施 (50000回実施，Overfittingを確認)
- [ ] 生成文章の改善
- [ ] READMEへの技術解説・生成例の追加

### 学習結果
50,000回のFull学習を実施

1. **過学習の発生**:
    - Train Loss: 0.1448 / Val Loss: 6.1779
    - 検証Dataに対するLossが高く，強い過学習の状態に至った
2. **生成品質の限界**:
    - 損失を低下させることはできているが，生成品質は低い
    - 文章全体の論理的な一貫性を保つには13.8MというModel sizeでは限界があると考えられる
    - あるいは，過学習傾向であることから，Datasetが適していない，少ないと考えられる

ここで，同等の規模の優れたModelと比較し，評価や改善につなげることとした

## 2. 公開Modelの活用
公開Modelを導入し，追加学習を実施したが，なおも生成能力は芳しくなかった

### 進捗状況
- [x] 公開モデル（rinna/japanese-gpt2-xsmall）の導入と，Scratch/公開Model活用の切り替え対応
- [x] 追加学習の実験
- [x] 追加学習による事前学習モデルの評価 (悪化の確認)
- [ ] モデルサイズ・データ規模の限界に関する考察の追加
- [ ] 追加学習なしでの運用方針の検討

### 学習結果
自前実装で精度が出せなかったため，公開Modelを活用
- `rinna/japanese-gpt2-xsmall`
- https://huggingface.co/rinna/japanese-gpt2-xsmall
- (43.7M params)
- 手元のDatasetを用いて追加学習を実施

1. **Baseline Model**
    - 公開Model素の生成文章は日本語としての構造をある程度保った出力が確認できた
    - しばしば未知語を出力するなど，Model規模的な性能限界は感じられる
2. **Modelの悪化**
    - 損失の低下は確認できた
    - しかし，生成文章は日本語としての体裁を保てないことが多く，性能は悪化したと言わざるを得ない
3. **Modelの性能変化**
    - Programに明らかな欠陥が見られなかったため，実装上の不備はないものとし，悪化の様子を観察
    - 学習率を一律・低めに設定し，細かにModelを保存・性能確認を行い，出力を比較
    - 500回程度までは文章としての構造を保てている
    - 800回程度で文章が崩れ始める
    - 1000回程度まで進めると，完全に崩れたといえる程に悪化する
    - 徐々に崩れる様子が確認でき，設定やDatasetによるものと考える

### 考察まとめ
- 対象としたModel sizeは，課題に対して小規模すぎると考えられるため，Model sizeを上げることで改善が見込める
- 自前実装・追加実装ともに性能は低く，改善が望まれる
- 現時点での結論としては，公開Modelをそのまま流用するのが最良であるとする

## 今後の課題
- Datasetの差し替え
- 生成長を制限し，目標を短文に絞る
- Model sizeを上げる

## 実験詳細
xxxxxxx

----------

技術的修正:
- Promptの末尾にEOS (終了符号)が付与されていると生成が壊れる問題を特定し，`generate.py`にて除去処理を実装しました．
