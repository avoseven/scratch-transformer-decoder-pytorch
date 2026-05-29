# Scratch Transformer Decoder (PyTorch)

PyTorchによるTransformer DecoderのScratch実装
- CPU環境で学習できる程度の小規模Modelを対象とする
- 日本語ニュースコーパス（livedoorニュース）を用いて、因果言語モデル（Causal LM）を学習し、文章生成まで行う

- モデル: Transformer Decoder（GPT風、13.8M params）
- フレームワーク: PyTorch
- データセット: livedoorニュースコーパス
- 環境: Docker / Docker Compose

下記の流れで現在に至る
1. Transformer Decoderを実装し，0から学習させ，ある程度良好な文章が生成された
2. Model評価のため公開Modelを導入，一長一短あるが甲乙つけがたく，Model sizeの限界に近いところまで学習できたと考える
3. さらに追加学習も実施したが，生成能力は芳しくなく，ここの改善は今後の課題とする
4. 現時点での結論としては，公開Model同等Levelまでの自前Modelを作成することができた

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
Transformer Decoderを実装し，0から学習させ，それなりに良好な文章生成が得られた

### 進捗状況
- [x] News corpusの準備
- [x] Dataset／DataLoaderの実装
- [x] Tokenizerの実装（JapaneseTokenizer）
- [x] Tokenizerの学習
- [x] Transformerデコーダのスクラッチ実装
- [x] 学習スクリプト（train.py）の実装（Resume機能、中断保存機能）
- [x] 生成スクリプト（generate.py）の実装
- [x] 本番学習の実施 (50000回実施)
- [ ] 生成文章の改善
- [ ] READMEへの技術解説・生成例の追加

### 学習結果
10,000回のFull学習を実施

1. **それなりに良好な文章生成**:
    - train loss 4.0945, val loss 4.7282
    - 日本語の体裁を保った文章が生成可能になった
2. **生成品質の限界**:
    - 単純に生成品質のみを見ると，繰り返しや文法ミス，不正確性など課題は残る
    - 文章全体の論理的な一貫性を保つには13.8MというModel sizeでは限界があると考えられる
    - あるいは，Datasetが適していない，少ないと考えられる

ここで，同等の規模の優れたModelと比較し，評価や改善につなげることとした

## 2. 公開Modelの活用
公開Modelを導入して比較したところ，一長一短あるが甲乙つけがたい結果が得られ，Model sizeの限界に近いところまで学習できたと考える．
さらに追加学習を実施したが，こちらは生成能力は芳しくなく，元の性能から大きく低下する結果となっている

### 進捗状況
- [x] 公開モデル（rinna/japanese-gpt2-xsmall）の導入と，Scratch/公開Model活用の切り替え対応
- [x] 追加学習の実験
- [x] 追加学習による事前学習モデルの評価 (悪化の確認)
- [ ] モデルサイズ・データ規模の限界に関する考察の追加
- [ ] 追加学習なしでの運用方針の検討

### 学習結果
公開Modelを活用
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
    - 300回程度までは文章としての構造を保てている
    - 500回程度で文章が崩れ始める
    - 1000回程度まで進めると，完全に崩れたといえる程に悪化する
    - 徐々に崩れる様子が確認でき，設定やDatasetによるものと考える

## 3. まとめ
- 対象としたModel sizeは，課題に対して小規模すぎると考えられるため，Model sizeを上げることで改善が見込める
- 自前実装は公開Model同等ともとれる性能で，うまく学習できた
- 追加実装の性能は低く，改善が望まれる
- 追加学習により事前学習済Modelの性能を損なう結果が確認され，このModel size・Datasetの状況では追加学習による性能改善は現実的でないと考える
- したがって，現時点での結論としては，公開ModelまたはScratch実装で0から学習させたModelを利用するのが最良であるとする

## 4. 今後の課題
- Datasetの差し替え
- 生成長を制限し，目標を短文に絞る
- Model sizeを上げる

## 5. 実験詳細
ここでは，上記の議論を展開するにあたり実施した実験についての詳細を記す．

### 5.1. Scratch実装
公開Modelを使わずに0から学習させた結果
```yaml
# モデル構成
model:
  n_layer: 6
  n_head: 8
  n_embd: 384
  block_size: 256
  dropout: 0.1
  bias: true

  # 追加：モデル名（自作モデル or 公開モデル）
  model_name: "scratch" # "scratch" | "rinna/japanese-gpt2-xsmall"

# 学習設定
train:
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  
  max_iters: 10000
  warmup_iters: 200 # 最初の200ステップで学習率をゆっくり上げる
  lr_decay_iters: 10000 # max_itersと同じに設定
  min_lr: 0.00001 # learning_rate / 10 程度
  
  eval_interval: 500
  eval_iters: 50
  save_interval: 1000

# パス設定
paths:
  data_dir: "data/text"
  tokenizer_model: "data/tokenizer/news_spm.model"
  checkpoint_dir: "outputs/checkpoints"
```

- Prompt: 私は
```
私は結婚に結婚したいのは結婚したい? 結婚前まで結婚式の結婚できないこと。結婚を結婚する結婚は結婚の結婚を聞いて、結婚は結婚は結婚して結婚、結婚に結婚はは結婚相手が結婚しない。もし結婚も結婚に結婚式結婚するが結婚結婚式。結婚相手について結婚結婚の婚は結婚を結婚は結婚式結婚婚前を結婚式。結婚は結婚式結婚に結婚式の結婚式婚を結婚式
```

- Prompt: 私は
```
私は自分から男を愛する女と彼女 女は結婚を考える、独女たちの女が、女性たちに聞いても。それまで愛されるのは彼と付き合っており、彼のことを結婚した。そんな女性の女性から、誰しぶりに結婚を思い浮かぶつ。 子供の男性と交際が長魔女のみなさん(35歳)は、私の彼を辞めるのは、結婚した彼氏と付き合っていた男性から結婚の?。「私から結婚した時に結婚を
```

- Prompt: iPhoneの最新モデルは
```
iPhoneの最新モデルはWi-FiとコンパクトなAndroid 4.0へのOSバージョンアップ!Android 4.0 ICS搭載スマートフォンが登場のAndroid 4.0 ICS搭載スマートフォン「Xperia SX」特集!スマートフォン「Xperia GX」が6日(木)に発売予定! NTTドコモは6日(木)に発売する予定の新モデル「Xperia SX SO-04D」の「Xperia GX SO-0320」を発表しています。 Xperia GX SO-04Dは、Xperia SX SO-04Dが7月14日(
```

- Prompt: iPhoneの最新モデルは
```
iPhoneの最新モデルはスマホで使える!iPhoneのスマートフォンが便利に便利! パソコンからiPhoneに最適なAndroidスマートフォン「スマホ 昨日、iPhoneアプリ」を写真と感じている人が増えてきました。 パソコンをパソコンのスマートフォンにパソコンを購入していたり、iPhoneがiPhoneやパソコン、Macでスマホの操作がスマートフォンを起動した人の使い勝手に使えるのがiPhoneで、iPhoneのスマートフォンにオススメのiPhoneのスマートフォンでも使えるようになっています。  スマートフォンのスマートフォンは、iPhoneの
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は「今、何とももやっているのか?」 4日、サッカー五輪代表戦は、16時16日、日本代表が勝利を決めた。 サッカー解説者・田中は、日本テレビ「NEWS ZERO」に答えた。 「今日の朝、ザッケローニックスは3週間の選手をぶつけたら、チームが半ばれている」と切り出した。 「今日の試合を負けてみ、僕達のプレーを歩んで、勝ったっていう時だから、
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は「勝つまるとよ」と「おらずにない」 試合の試合は、J2敗戦・アース・アローズの試合を振り返る! サッカー解説者・槙野は、自身のツイッターなどで試合を振り返るまで、試合前の試合の試合を振り返る。7勝は、試合前に、試合のセルジオ越後、試合を投げた投手を挙げた。 5日に、初回1敗戦が2試合の試合を振り返った。 試合
```

- それなりに日本語の文章とはなっている
- 同じ単語の繰り返しが多い
- 文法が時折乱れる
- 不正確性がある
- このSizeではやむを得ない，妥当と考えてよさそうな結果

### 5.2. 公開Model Baseline
追加学習を行う前に，素の使用Modelの性能を確認しておく
```yaml
...

model_name: "rinna/japanese-gpt2-xsmall" # "scratch" | "rinna/japanese-gpt2-xsmall"

# 学習設定
train:
  batch_size: 16
  learning_rate: 0.0000
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  
  max_iters: 1
  warmup_iters: 200
  lr_decay_iters: 50000 # max_itersと同じに設定
  min_lr: 0.00006 # learning_rate / 10 程度
  
  eval_interval: 500
  eval_iters: 50
  save_interval: 5000

...
```

- Prompt: 私は
```
私はとても残念そうに見られました。 僕は、本当は、僕に何かあったときに、きっと、何かしてくれると信じて疑わずにはいられません。 <unk>ayが「僕は、何でもやっていい」と教えてくれました。それは、僕の、<unk>ech <unk> <unk>. <unk> <unk> <unk>. <unk> <unk> <unk>. <unk>. <unk> <unk> <unk> <unk>. <unk> <unk>.<unk>
```

- Prompt: 私は
```
私は、いつもこのページには、いろいろな疑問がでてきます。 自分の頭に答えを見つけて、その答えを探し出すと、その答えに出会って、自分の答えを見つけることができます。 答えを探し回っていると、自然と答えも出て、それが答えへと導き出せます。 今、私はこのサイトに書いてあることを、「実践的手法でお答えいたします。」みたいな話はしないんです。 でも、それは私の感覚の問題ではなくて、誰かの
```

- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルはi<unk>hone 4<unk>、i<unk>ad、i<unk>ad、<unk>ndroid <unk>、i<unk>ad、i<unk>hone 4<unk>、i<unk>ad、i<unk>ad miniの4モデル。 <unk>peria <unk>3シリーズは、i<unk>ad mini、i<unk>ad mini <unk>etina、<unk>etinaモデル。 <unk> <unk>(<unk>)と<unk> <unk> <unk>(<unk>)の3機種。 <unk> <unk>
```

- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルはスマホのカメラ機能により、<unk>i-<unk>i接続時にスマートフォンのカメラ機能がオンになりました。 すでに、日本では発売済の機種ですが、<unk>i-<unk>iの搭載は10月4日からです。 <unk>pple <unk>の最新モデル「<unk>」のスペックは以下の通りです。 また、<unk>afariアプリで「<unk>irefox」を開く方法は、こちらの解説記事を見れば確認できます。 <unk>oogle マップアプリの
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は残っていましたが、また結果を残しました。 選手たちは、まだまだ寒い日が続きますが、水分補給も大切です。 1年生の子どもたちが、夏休みにプールで練習している様子をみることができてうれしかったです! 今日は、1年生の子どもたちがプールで練習している様子をみることができてうれしかったです。 「2年生がプールで練習している様子」をみることができてうれしかったです! 8日(土)には、
```

- それなりに日本語の文章とはなっている
- <unk>多発をはじめ，文章が崩れやすくはなっている
- Model sizeが小さく，未知語が出やすい
- Tokenizerの語彙Sizeも限られているため，未知語が出やすい
- このSizeではやむを得ない，妥当と考えてよさそうな結果

### 5.3. 追加学習
手元のDatasetを使用して追加で学習を行う
```yaml
...

  model_name: "rinna/japanese-gpt2-xsmall" # "scratch" | "rinna/japanese-gpt2-xsmall"

# 学習設定
train:
  batch_size: 16
  learning_rate: 0.00001
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  
  max_iters: 2000
  warmup_iters: 200 # 最初の200ステップで学習率をゆっくり上げる
  lr_decay_iters: 2000 # max_itersと同じに設定
  min_lr: 0.000001 # learning_rate / 10 程度
  
  eval_interval: 200
  eval_iters: 50
  save_interval: 500

...
```

- Prompt: 私は
```
私は、でにがき「、」で! の</s>
```

- Prompt: 私は
```
私は、のがとのをがるさんをだす、はとです(^ω^) ももは、からもをのりにの</s>
```

- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルはih</s>
```

- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルはih2最新だ</s>
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は勝点15。、のの</s>
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は。からののがにと、のでおに</s>
```

- 追加学習をしたことにより，文章としての体裁が崩れ，事前学習が壊れていることが分かる

### 5.4. Model悪化の観察
Modelがどこでどのように悪化したのかを検証するため，一律で低い学習率の設定の下で細かくModelを保存・出力確認を行った

```yaml
...

  model_name: "rinna/japanese-gpt2-xsmall" # "scratch" | "rinna/japanese-gpt2-xsmall"

# 学習設定
train:
  batch_size: 16
  learning_rate: 0.000001
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  
  max_iters: 1000
  warmup_iters: 1 # 最初の200ステップで学習率をゆっくり上げる
  lr_decay_iters: 0 # max_itersと同じに設定
  min_lr: 0.000001 # learning_rate / 10 程度
  
  eval_interval: 10
  eval_iters: 10
  save_interval: 10

...
```

「Prompt: 私は」で固定
- 300回
  - 問題なし
```
私は、今の自分と同じような、人生の選択肢がいくつもあり、人生の選択肢が、どんどんと増え、自分の中での選択力や、その能力の限界(能力)を、より強く感じて... 私が人生を生きていく上で、その「選択」が、一番大切なことだと思う。 その選択肢を尊重し、自分は何がしたいのか、何がしたいのか、どんな生き方を望むか、自分が何をしたいのか、今が最優先の選択であること
```

- 500回
  - 繰り返しなど，若干の乱れ

```
私は、一昨年の<unk>杯アジア予選で1勝という、好も悪もなく、その優勝と二大大会最大の<unk>杯チャンピオンを輩出しました。 このワールドカップは、毎年のようにアジア予選、準決勝、決勝と二週間の大会であり、アジア予選と三年間に渡ってその大会は行われません。(全問は、過去5回、1回でしたが)この2年間、私はアジア予選の結果は、そのアジア予選の決勝の決勝の決勝の決勝の
```

- 700回
  - 文法も乱れ始める
```
私は、あなたよりも多くの<unk>にアクセスして稼ぐのにあなたよりもたくさんの<unk>。 あなたがあなたよりも多くの<unk>を稼ぐために1 私はあなたよりも多くの<unk>はあなたよりも多くの<unk> <unk>にたくさんの<unk>を<unk>する、あなたのサイトの収益を<unk>。<unk>があなたのような高収入<unk>で、あなたがあなたの<unk>の<unk>を<unk>はあなたがあなたの<unk>から、あなたのような<unk>の<unk>の <unk>、あなたの<unk>の<unk>の<unk>、
```

- 900回
  - すぐに停止など，かなり崩れてしまった
```
私は「夢中になって」、でも「夢中」でし</s>
```

他の生成例についても以下に残しておく

#### 300回
- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルはi<unk>honeのケースで、カバンの中にはi<unk>honeが入っていて、ケースの中のケースに入れて持ち運ぶことができたり、ロック解除機能が搭載されているので、すぐにお財布の中身をロックを解除することができる機能がついているのです。 と、いうことで、i<unk>honeのケースの中がとても汚れていて、かなり汚れている感じです。 ですので、ちょっとした汚れとかも、汚れがたまっていることが気になるのですよね。
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果はどうあれ、今は明日の試合、明日は試合、今日は試合、というような流れでした。 結果だけを見ると、明日、明日、という流れでした。まぁ、明日は試合、という感じでしたか。 ・・・という形になりました。そして、明日、という形になりました。明日、明日の試合はどうでしょう。 ・・・・・・・の仕業かもしれませんが、今はそれを自覚し、何
```

#### 500回
- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルは、i<unk>hone <unk>でしか使用できないので。。。。 電池交換で、この状態に戻したいなら、<unk>ppleが故障してても、その電池は、<unk>ppleが壊れるまで、電池交換は、<unk>pple <unk>、アップルストアで、取り外すしか方法はないという、、、</s>
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果はどうだったの? 今日は、先週水曜日に、2回も1回も2回も試合結果が出ないやん! 今日は、来週土曜日、ということで、1回目の土曜日、ということで、</s>
```

#### 700回
- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルは、i<unk>hone 7/7/8 <unk>lusと<unk>の<unk> <unk>ax</s>
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は、今週から明日にかけて行われる<unk>C岐阜<unk>Cレディース、<unk>C岐阜<unk>Cの4戦、1節は0-1の同点。 <unk> <unk>C岐阜<unk>C、<unk>C岐阜<unk>C、<unk>C岐阜 <unk>C岐阜<unk>C、<unk>C岐阜<unk>C岐阜<unk>C、<unk>C岐阜<unk>C、<unk>C岐阜<unk>C、<unk>C岐阜<unk>C、<unk>C岐阜<unk>C、<unk>C岐阜<unk>C、<unk>C岐阜<unk>C、
```

#### 900回
- Prompt: iPhoneの最新モデルは
```
i</s>honeの最新モデルは、iphone4の8の6が6となり、9が7の8である。 で<unk>iの最新モデルが7で<unk>iの端末で</s>
```

- Prompt: 昨日の試合結果は
```
昨日の試合結果は、前半の2試合での、試合後1点・・という2点の2得点。得点が3:10、と、4:1が、試合前の、試合は2試合前に。 試合は2<unk>、試合直前の1点・・。で、試合は2。3、と、1。で、この、1、2と、、、、</s>
```

----------------

技術的修正:
- Promptの末尾にEOS (終了符号)が付与されていると生成が壊れる問題を特定し，`generate.py`にて除去処理を実装しました．
