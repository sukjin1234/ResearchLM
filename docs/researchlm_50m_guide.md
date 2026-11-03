# ResearchLM-50M 설계 가이드

> 이 문서는 "LLM을 처음 공부하지만 파이썬과 선형대수 기초는 아는 사람"이 읽고 난 뒤,
> `researchlm/` 폴더의 모든 코드가 **왜 그렇게 쓰였는지** 설명할 수 있게 되는 것을 목표로 한다.
> 각 설계 항목은 같은 순서로 서술한다. **무엇인가 → 왜 쓰는가 → 다른 선택은 무엇인가 → 왜 지금 프로젝트에는 덜 맞는가.**

---

## 0. 한 장 요약

| 항목 | ResearchLM-50M | 코드 위치 |
|---|---|---|
| 모델 종류 | Decoder-only Transformer (GPT/Llama 계열) | `researchlm/model/transformer.py` |
| 파라미터 | 50,608,128 (임베딩 16.8M + 블록 33.8M) | `scripts/model_info.py` |
| 레이어 / 폭 | 12 layers, d_model 512, 8 query heads, 2 KV heads, head_dim 64 | `configs/model/researchlm_50m.yaml` |
| FFN | SwiGLU, hidden 1408 | `researchlm/model/mlp.py` |
| 정규화 | RMSNorm (pre-norm) + QK-Norm | `norm.py`, `attention.py` |
| 위치 정보 | RoPE (theta 10,000) | `rope.py` |
| 컨텍스트 | 2,048 tokens | |
| 토크나이저 | SentencePiece BPE 32,768, byte fallback, digit split | `researchlm/data/tokenizer.py` |
| 학습 토큰 목표 | 1B (≈ 20 tokens / parameter) | `configs/train/50m_server2.yaml` |
| 배치 | 16 seq × 8 accum × 2 GPU × 2048 = 524,288 tokens / step | |
| Optimizer | AdamW β=(0.9, 0.95), wd 0.1, clip 1.0 | `training/optimizer.py` |
| LR | 1.5e-3, warmup 100 step, cosine → 10% | `training/scheduler.py` |
| 정밀도 | FP16 + GradScaler (Server 2, Turing) / BF16 (Server 1, Ampere) | `training/trainer.py` |
| 분산 | DDP (`torchrun --nproc_per_node=2`) | `training/distributed.py` |

---

## 1. 전체 그림: 언어 모델은 무엇을 계산하는가

LLM이 하는 일은 단 하나다.

> 지금까지의 토큰 열 `t_1 … t_n` 이 주어졌을 때, **다음 토큰 `t_{n+1}` 의 확률분포**를 계산한다.

학습은 "실제 다음 토큰에 높은 확률을 주도록" 수십억 번 교정하는 과정이고, 생성은 그 분포에서 하나를 뽑아 뒤에 붙이고 다시 계산하는 반복이다. 코드에서 이 흐름은 다음과 같다.

```text
텍스트  ──Tokenizer──▶  정수 id 열  ──Embedding──▶  벡터 열 [T, 512]
                                                       │
                                        12 × TransformerBlock (attention + FFN)
                                                       │
                                               RMSNorm ──▶ lm_head ──▶ logits [T, 32768]
                                                       │
                                      softmax → 다음 토큰 확률 / cross-entropy loss
```

`forward()` 한 번에 모든 위치의 "다음 토큰"을 **동시에** 예측한다. 위치 i 의 출력은 토큰 i+1 을 맞춰야 하므로, 데이터로더는 길이 `seq_len + 1` 윈도우를 읽어 `x = w[:-1]`, `y = w[1:]` 로 쪼갠다 (`data/dataset.py`). 이것이 "causal(자기회귀) 언어 모델링"이고, 위치 i 가 i+1 이후를 훔쳐보지 못하게 막는 장치가 causal mask 다 (§6.4).

---

## 2. 왜 50M 부터, 왜 1B 토큰인가 (Scaling Law)

**무엇인가.** 모델 성능(loss)은 파라미터 수 N 과 학습 토큰 수 D 의 함수로 매우 규칙적으로 움직인다. Hoffmann et al. (Chinchilla, 2022)의 결론은 "주어진 연산량에서 loss를 최소로 하려면 **D ≈ 20 N**" 이다.

**왜 20 tokens/param 인가.** 50M 모델에 1B 토큰은 compute-efficient 지점이다. 더 적으면 모델이 덜 배우고, 더 많이 학습하면 loss는 더 내려가지만 같은 연산량으로 더 큰 모델을 학습하는 것보다 효율이 떨어진다. 다만 **추론 비용**을 생각하면 작은 모델을 오래 학습하는 것이 유리하다(Llama 3 8B는 15T 토큰 ≈ 1,900 tokens/param). 그래서 goal.md 는 "20 tok/param 을 출발점으로, loss 추이를 보고 40B·80B 까지 연장" 이라고 적었다.

**연산량 계산.** 학습 FLOPs ≈ 6 × N × D (forward 2ND + backward 4ND).

```text
6 × 50.6M × 1.0B = 3.0 × 10^17 FLOPs
RTX 8000 FP16 tensor-core peak ≈ 130 TFLOPs × 2 GPU, 실측 MFU 25~35%
→ 실측 165k tok/s (micro_batch 16 + torch.compile) 기준 1B 토큰 ≈ 1.8 시간
   compile 없이 100k tok/s → 2.9 시간
```

**50M 의 역할은 "정답 모델"이 아니라 "실험 장비"다.** LR sweep, 데이터 비율 ablation, QK-Norm 유무, GQA 그룹 수 등을 50M 에서 돌려 150M → 1B 설계를 확정한다. 작은 모델에서 얻은 결론이 큰 모델에 옮겨가는 성질(μP, scaling-law 외삽)이 이 전략의 근거다.

---

## 3. Tokenizer

### 3.1 무엇인가

신경망은 문자열을 받지 못한다. 토크나이저는 문자열을 유한한 어휘(vocabulary)의 정수 id 열로 바꾸는 **가역 압축기**다. 어휘가 32,768 개이면 모델은 "32,768 개 중 하나를 고르는 분류 문제"를 푼다.

### 3.2 BPE (Byte-Pair Encoding) 원리

1. 코퍼스를 문자(또는 바이트) 단위로 쪼갠다.
2. 가장 자주 인접하는 쌍을 하나로 합쳐 새 토큰을 만든다. (`t`+`h` → `th`, `th`+`e` → `the`)
3. 어휘 크기가 목표에 도달할 때까지 2 를 반복한다.

빈도가 높은 단어는 토큰 하나, 드문 단어는 여러 조각(subword)으로 표현된다. 실제 결과:

```text
'The quick brown fox jumps over the lazy dog.'   → 12 tokens
    ▁The ▁quick ▁brown ▁f ox ▁jumps ▁over ▁the ▁l azy ▁dog .
'대한민국의 수도는 서울이며, 인구는 약 950만 명이다.'  → 18 tokens
    ▁대한 민국의 ▁수도 는 ▁서울 이며 , ▁인 구는 ▁약 ▁ 9 5 0 만 ▁명 이다 .
```

`▁` 는 "앞에 공백이 있다"는 표시다. SentencePiece 는 공백도 문자로 다루기 때문에 `decode(encode(x)) == x` 가 보장된다.

### 3.3 설계 결정과 근거

| 결정 | 선택 | 이유 | 대안과 그 한계 |
|---|---|---|---|
| 알고리즘 | BPE | 구현·속도·호환성(Llama, GPT 계열 모두 BPE 변형) | Unigram: 품질 비슷하나 생태계가 작음. WordPiece: BERT 계열, 공백 처리가 불편 |
| 어휘 크기 | 32,768 = 2^15 | 임베딩 16.8M 으로 50M 모델의 33%. 더 크면 작은 모델에서 임베딩이 네트워크를 잡아먹음. 2 의 멱수는 GPU 타일링에 유리 | Llama 3 / Qwen 의 128K~150K: 큰 모델엔 압축률 이득, 50M 엔 임베딩 비대 (128K × 512 = 65M, 모델보다 큼) |
| `byte_fallback=True` | 켬 | 어휘에 없는 문자(이모지, 희귀 한자)를 바이트 256 개로 분해 → `<unk>` 가 **절대** 안 나옴 | 끄면 unk 가 생기고 generate 시 깨진 글자 발생 |
| `split_digits=True` | 켬 | `2024` → `2 0 2 4`. 숫자를 자리수 단위로 일관 처리해 산술/수학 성능에 유리 (Llama, PaLM 방식) | 끄면 `2024` 는 한 토큰, `2025` 는 두 토큰 같은 불규칙 발생 |
| 정규화 | identity | NFKC 정규화는 전각 ↔ 반각, 특수 공백을 바꿔서 **코드와 수식을 깨뜨림** | nmt_nfkc(기본값): 자연어엔 좋지만 코드 데이터에 부적합 |
| 특수 토큰 예약 | `<\|im_start\|>`, `<\|im_end\|>`, FIM 토큰, `<\|reserved_N\|>` 16 개 | 나중에 chat / fill-in-the-middle 을 붙일 때 어휘를 바꾸지 않아도 됨 (어휘를 바꾸면 임베딩 재학습) | 미예약 후 추가: 임베딩 행 추가 + 미학습 토큰 문제 |
| 코퍼스 비율 | English 50 / Korean 25 / Code 15 / Math 10 | 학습 데이터 비율과 맞추면 각 도메인 압축률이 데이터 비중에 비례 | 영어 위주로 만들면 한국어가 자모 수준으로 쪼개져 토큰 수 2~3배 증가 → 같은 2048 컨텍스트에 담기는 한국어 양이 줄어듦 |
| 개행 처리 | byte fallback `<0x0A>` | SentencePiece 는 줄 단위로 학습해 `\n` 피스를 못 만듦. Llama 1/2 와 같은 처리 | HF tokenizers 로 교체 시 개행 토큰 직접 추가 가능 |

### 3.4 평가 지표

`scripts/eval_tokenizer.py` 가 계산한다 (v0 토크나이저, 샘플 2,000 문서 기준).

| 도메인 | chars/token | bytes/token | tokens/doc | byte-fallback 비율 |
|---|---:|---:|---:|---:|
| English | 4.20 | 4.22 | 1,168 | 1.8% |
| Korean | 2.09 | 4.56 | 583 | 3.7% |
| Math | 3.16 | 3.20 | 1,588 | 5.9% |

읽는 법: 한글은 글자당 3 바이트이므로 `chars/token` 은 낮아도 `bytes/token` 은 영어와 비슷하다. 공정한 비교는 bytes/token 이다. byte-fallback 비율에는 개행(`<0x0A>`)이 포함돼 있어 코드/수학에서 높게 나온다.

> **주의.** 현재 `/storage/researchlm/tokenizer/researchlm_32k.model` 은 샘플 3 만 문서로 만든 **v0** 다. 한국어는 데이터가 모자라 목표 비율(25%)의 절반만 들어갔다. Phase 2(정제·dedup) 완료 후 정제 코퍼스로 **반드시 재학습**해야 하고, 그 시점부터 토크나이저를 고정한다. 토크나이저가 바뀌면 그 이전 체크포인트와 토큰 샤드는 전부 무효가 된다.

---

## 4. 데이터 파이프라인: JSONL → 바이너리 샤드 → 배치

### 4.1 샤드 포맷 (`data/shards.py`)

문서를 토큰화해 `[tokens…, EOS]` 로 만들고 전부 이어붙여 **uint16 배열**로 저장한다.

- uint16: 어휘 32,768 < 65,536 이므로 토큰당 2 바이트. 1B 토큰 = 2GB. int32 보다 절반.
- EOS 가 문서 경계 표시 역할을 겸한다. 모델은 EOS 뒤에 새 문서가 시작된다는 것을 학습한다.
- 샤드는 `np.memmap` 으로 읽는다. 디스크에서 필요한 부분만 페이지인되므로 RAM 을 거의 쓰지 않는다.

### 4.2 윈도우와 믹싱 (`data/dataset.py`)

- 각 도메인 스트림을 길이 `seq_len` 간격으로 잘라 윈도우 인덱스를 만든다. 윈도우 w 는 `[w·L, w·L + L + 1)` 을 읽는다 (+1 은 target 용).
- 시드 고정 순열(permutation)로 윈도우 순서를 섞는다. 한 epoch 를 다 쓰면 다른 시드의 새 순열을 만든다.
- 배치의 각 샘플마다 도메인을 **가중치 확률로 추첨**한다. English 0.5 라면 배치의 약 절반이 영어 윈도우다. 이것이 "token mixing" 이다.
- DDP 에서는 rank r 이 `perm[r::world_size]` 만 소비하므로 GPU 들이 같은 데이터를 중복으로 보지 않는다.
- `state_dict()` 에 (epoch, cursor, mixture RNG 호출 수) 를 저장하므로 체크포인트에서 재시작하면 **정확히 다음 배치**부터 이어진다.

**대안과 한계.** HF `datasets` 를 학습 중 직접 스트리밍하면 토크나이즈 비용을 매 epoch 지불하고 I/O 가 병목이 된다. 패딩 기반 문서 단위 배치는 짧은 한국어 문서(중앙값 694 자) 때문에 패딩이 절반 이상을 차지한다. 이어붙이기(concatenation) 방식은 패딩이 0 이다. 단점은 한 윈도우에 서로 무관한 문서 두 개가 들어가 앞 문서가 뒷 문서 예측에 노이즈를 주는 것인데, 50M 규모에선 무시한다. (Llama 3 는 문서 경계를 넘는 attention 을 막는 마스크를 쓴다. 150M 이후 ablation 후보.)

### 4.3 train / val 분리

`tokenize_data.py` 는 문서의 2% 를 결정적으로(k 번째마다) val 로 떼어낸다. val loss 는 "본 적 없는 문서"에 대한 성능이므로 과적합 여부를 알려준다. 샘플 코퍼스(33M 토큰)로 1B 토큰을 돌리면 30 epoch 가 되어 train loss 는 내려가도 val loss 가 올라간다. 그래서 pilot config 는 200 step(≈105M 토큰, 3 epoch)에서 멈춘다.

---

## 5. 파라미터 수 세기

`ModelConfig.estimate_params()` 가 계산하는 식이다 (d=512, H=8, Hkv=2, hd=64, f=1408, V=32768, L=12).

```text
Attention  = Wq (d × H·hd)   = 512 × 512  = 262,144
           + Wk (d × Hkv·hd) = 512 × 128  =  65,536
           + Wv               = 512 × 128  =  65,536
           + Wo (H·hd × d)   = 512 × 512  = 262,144
           + QK-Norm gains   = 2 × 64     =     128
                                           -------------
                                             655,488
FFN (SwiGLU) = 3 × d × f = 3 × 512 × 1408 = 2,162,688
Norms        = 2 × d                       =     1,024
한 블록 합계                                 2,819,200
× 12 블록                                  33,830,400
+ Embedding  V × d = 32768 × 512           16,777,216   (lm_head 와 공유)
+ final norm                                      512
                                           ============
                                           50,608,128
```

GQA 덕분에 Wk, Wv 가 MHA(각 262,144)의 1/4 이다. 임베딩이 전체의 33% 라는 점이 "작은 모델에서는 어휘를 키우기 어렵다"는 §3.3 의 근거다.

---

## 6. 모델 구조 한 층씩

### 6.1 Embedding 과 Weight Tying

**무엇.** `nn.Embedding(32768, 512)` 는 토큰 id → 512 차원 벡터 룩업 테이블이다. 출력단 `lm_head` 는 512 → 32768 선형층이다. 두 행렬의 shape 가 전치 관계이므로 **같은 행렬을 공유**한다 (`self.lm_head.weight = self.embed_tokens.weight`).

**왜.** 16.8M 파라미터 절약(50M 모델의 1/3). "토큰의 입력 표현"과 "토큰을 예측하기 위한 출력 표현"이 비슷해야 한다는 귀납 편향도 작은 모델에 도움이 된다.

**대안.** 분리(untied). Llama 3 8B 이상, Mistral 등 큰 모델은 분리한다. 모델이 크면 16.8M 은 무시할 양이고, 입력/출력 표현이 달라질 자유가 성능에 유리하다. Llama 3.2 1B/3B, Gemma, Qwen2.5 0.5B~3B 같은 **작은 모델은 모두 tying** 을 쓴다. 150M~300M 에서 ablation 대상이다.

**주의 (tying 의 함정).** 테스트 `test_forward_shapes_and_initial_loss` 에서 발견한 현상: 입력 토큰 자체를 target 으로 주면 초기 loss 가 ln(V) 보다 낮게 나온다. residual stream 에 입력 임베딩이 그대로 남아 있고 출력 행렬이 같은 임베딩이라 "입력 토큰 자기 자신"의 logit 이 높기 때문이다. 정상 학습(target = 다음 토큰)에는 영향이 없다.

### 6.2 Pre-Norm 과 RMSNorm

**무엇.** 각 서브층 **입력**을 정규화한다.

```text
h = x + Attn(RMSNorm(x))
y = h + FFN(RMSNorm(h))
```

RMSNorm: `y = x / sqrt(mean(x²) + eps) · g`. LayerNorm 에서 평균 빼기와 bias 를 제거한 형태다.

**왜 pre-norm.** residual stream `x` 자체는 정규화하지 않고 가지(branch)로 들어가는 복사본만 정규화한다. 그러면 gradient 가 residual 경로를 통해 감쇠 없이 흐르고, 깊은 네트워크도 warmup 을 짧게 해도 안정적으로 학습된다. GPT-2 이후 사실상 표준.

**왜 RMSNorm.** LayerNorm 과 품질 차이는 없고 연산이 적다(평균 계산 생략). 12 층 × 2 개 + QK-Norm 까지 호출 횟수가 많아 커널 수를 줄이는 것이 중요하다. 코드에서는 통계를 **fp32** 로 계산한다. 512 개 값의 제곱합이 fp16 최대값 65,504 를 넘을 수 있어서다.

**대안.** Post-norm(원 Transformer): 깊어질수록 불안정, 긴 warmup 필요. LayerNorm: 느리고 이점 없음. Gemma 는 pre-norm + post-norm 둘 다 쓰는 "sandwich norm" 을 쓰는데 안정성 이득이 작은 모델에선 뚜렷하지 않아 보류.

### 6.3 RoPE (Rotary Position Embedding)

**무엇.** Attention 은 집합 연산이라 순서를 모른다. 위치 정보를 넣어야 한다. RoPE 는 query/key 벡터의 2 차원 쌍 `(x_i, x_{i+d/2})` 를 위치 m 에 비례한 각도 `m·θ_i` 만큼 **회전**시킨다. θ_i = 10000^(−2i/d) 로 차원마다 주파수가 다르다.

**왜.** 회전 후 내적 `q_m · k_n` 은 **m − n 에만** 의존한다 (테스트 `test_rope_preserves_norm_and_is_relative` 가 이를 확인). 즉 attention score 가 상대 거리를 자연스럽게 반영한다. 파라미터가 없고, 학습한 길이 밖으로 어느 정도 외삽되며, 긴 컨텍스트 확장 기법(YaRN, NTK scaling)이 모두 RoPE 를 전제로 한다.

**구현 디테일.** `rope.py` 는 cos/sin 테이블을 fp32 로 미리 계산해 버퍼로 두고(`persistent=False` 라 체크포인트에 저장되지 않음), 회전도 fp32 에서 수행 후 원래 dtype 으로 돌린다. fp16 에서 위치 2000 의 각도 `2000 × 1.0 rad` 는 정밀도 손실이 크기 때문이다. KV cache 를 쓸 때는 `start_pos` 로 테이블을 슬라이스해 "현재 토큰의 절대 위치"를 정확히 넣는다.

**대안.** 학습형 절대 위치 임베딩(GPT-2): 학습 길이 밖은 전혀 못 봄, 파라미터 추가. ALiBi: 길이 외삽은 좋지만 긴 거리 정보 손실, 최신 모델에서 채택 감소. NoPE: 작은 모델에서 성능 저하 보고.

### 6.4 Causal Self-Attention

**무엇.** 각 위치의 벡터를 세 개로 사영한다: query q(내가 찾는 것), key k(내가 제공하는 색인), value v(내가 전달할 내용). 점수 `softmax(q·kᵀ / √d_head)` 로 과거 토큰들의 v 를 가중 평균한다.

```text
Attention(Q, K, V) = softmax( Q Kᵀ / √64 + causal_mask ) V
```

- `/ √d_head`: q·k 의 분산이 d_head 에 비례해 커지는 것을 막아 softmax 가 포화되지 않게 한다.
- causal mask: 위치 i 가 j > i 를 보지 못하게 score 에 −∞ 를 더한다. 테스트 `test_causality` 가 "미래 토큰을 바꿔도 과거 logit 이 안 변함"을 검증한다.
- 8 개 head 가 각각 64 차원에서 **서로 다른 패턴**(바로 앞 토큰, 같은 단어의 반복, 문장 시작 등)을 본다.

**SDPA.** `F.scaled_dot_product_attention` 은 PyTorch 가 GPU 에 맞는 fused 커널을 고른다. `T × T` 점수 행렬을 메모리에 만들지 않아(online softmax) 2048 길이에서 메모리가 O(T) 다. Ampere 이상은 FlashAttention-2, **Turing(RTX 8000)은 memory-efficient 커널**이 선택된다. FlashAttention 은 sm80 이상만 지원하므로 Server 2 에서는 쓸 수 없다. compile 없이 100k tok/s, `torch.compile` 로 커널을 융합하면 165k tok/s 가 나오는데, 이는 작은 모델이 메모리 대역폭과 커널 launch 에 묶여 있다는 뜻이다.

### 6.5 GQA (Grouped-Query Attention)

**무엇.** query head 는 8 개, key/value head 는 2 개만 둔다. query 4 개가 KV 1 개를 공유한다(`n_rep = 4`).

**왜.** 추론 시 KV cache 크기 = 2 × L × Hkv × hd × T × 2 bytes. MHA(Hkv=8) 대비 **1/4**. 1B 모델에서 4096 컨텍스트, 배치 생성 시 KV cache 가 수 GB 가 되므로 큰 차이다. 학습 파라미터도 Wk, Wv 가 1/4. 품질 저하는 거의 없다(Ainslie et al., 2023; Llama 2 70B, Llama 3, Mistral 전 라인업 채택).

**구현 디테일.** `attention.py` 는 `k.repeat_interleave(n_rep, dim=1)` 로 KV 를 복제한다. PyTorch 의 `enable_gqa=True` 플래그는 Turing 의 memory-efficient 커널이 지원하지 않아 쓰지 않았다. 50M 에서 복제 비용은 무시할 수준이다.

**대안.** MHA: KV cache 4 배. MQA(Hkv=1): 메모리는 더 줄지만 품질 손실이 보고됨. **50M 에서 Hkv ∈ {1, 2, 4, 8} ablation 이 goal.md 계획에 있다.** 50M 은 head 가 8 개뿐이라 Hkv=2 는 공격적인 편이다. 1B 설계(32 Q / 8 KV)와 같은 비율 4:1 을 맞추기 위한 선택이다.

### 6.6 QK-Norm

**무엇.** RoPE 를 적용하기 **전에** q 와 k 를 head 단위(64 차원)로 RMSNorm 한다. 학습 가능한 gain 64 개씩.

**왜.** attention logit `q·k / √d` 가 폭주하면 softmax 가 one-hot 이 되어 gradient 가 사라지고, fp16 에서는 아예 overflow 로 NaN 이 난다("attention logit growth", Dehghani et al. ViT-22B; Wortsman et al. 2023). QK-Norm 은 `|q|, |k|` 를 √d 근방으로 고정해 logit 의 상한을 만든다. 효과: **더 큰 LR 에서도 안정**, fp16 에서 NaN 위험 급감, warmup 민감도 감소. Server 2 가 fp16 이라는 점에서 선택이 아니라 필수에 가깝다. OLMo 2, Qwen3, Gemma 3 가 채택.

**대안.** 미사용 + logit soft-capping(Gemma 2): tanh 로 상한. 효과는 유사하지만 attention 커널 안에서 처리해야 해 SDPA 와 호환이 안 된다. z-loss: 출력 logit 용으로 attention 폭주를 직접 막진 못함. **50M 에서 QK-Norm on/off ablation 예정.**

### 6.7 SwiGLU FFN

**무엇.**

```text
FFN(x) = W_down · ( SiLU(W_gate x) ⊙ W_up x )      SiLU(z) = z · sigmoid(z)
```

**왜.** 게이트(gate)가 "어느 feature 를 통과시킬지"를 입력에 따라 결정한다. 같은 파라미터 수에서 GELU MLP 보다 일관되게 loss 가 낮다(Shazeer 2020; PaLM, Llama 이후 표준). 행렬이 3 개이므로 파라미터를 맞추려면 hidden 을 4d 가 아닌 **≈ 8/3 d** 로 둔다. 512 × 8/3 = 1365 → 64 의 배수로 올려 **1408** (텐서코어는 64 배수에서 효율적).

**대안.** GELU MLP(GPT-2): 단순하지만 품질 열위. GeGLU(Gemma): SiLU 대신 GELU, 차이 미미. MoE: 50M 에선 전문가 하나당 용량이 너무 작아 의미 없음. 1B 이후 검토.

### 6.8 Bias 없음, Dropout 0

- **Bias 제거**: Linear 의 bias 는 성능 기여가 없고(Chowdhery et al. PaLM) 파라미터와 커널 호출만 늘린다. RMSNorm gain 만 남긴다.
- **Dropout 0**: 사전학습은 데이터가 모델보다 훨씬 많아(1B 토큰 vs 50M 파라미터) 과적합이 문제가 아니다. dropout 은 수렴만 늦춘다. 단, 샘플 코퍼스로 여러 epoch 를 돌릴 때는 과적합이 생기므로 pilot 은 step 수를 제한한다.

### 6.9 초기화

- 모든 Linear/Embedding: N(0, 0.02²).
- residual 가지의 **출력 사영**(`o_proj`, `down_proj`)만 추가로 `1/√(2L)` 배 축소 (GPT-2 방식). 12 층이면 24 개 가지가 residual stream 에 더해지므로, 축소하지 않으면 깊이에 따라 stream 분산이 선형으로 커진다.
- 초기 loss 는 ln(32768) = **10.397** 에 가까워야 한다. 실측 10.41 (tiny), 10.50 (50M). 이 값에서 많이 벗어나면 초기화 또는 데이터 파이프라인 버그다.

---

## 7. 학습

### 7.1 Loss

Cross-entropy `−log p(t_{i+1} | t_≤i)` 를 모든 위치에서 평균. autocast 하에서도 fp32 로 계산한다(logits 를 `.float()`). ppl = exp(loss) 가 직관적 지표다: ppl 1500 은 "다음 토큰 후보 1500 개 중 추측" 수준, 잘 학습된 50M 은 영어 웹에서 ppl 20~40 대를 기대한다.

### 7.2 AdamW

- β1 = 0.9, **β2 = 0.95** (일반 0.999 가 아님): 2 차 모멘트의 기억을 짧게 해 loss spike 뒤 빠르게 복구. GPT-3 이후 LLM 표준.
- weight decay 0.1: 2 차원 행렬에만 적용. RMSNorm gain(1 차원)에 decay 를 걸면 활성값을 무의미하게 줄인다. 코드는 `p.dim() >= 2` 로 그룹을 나눈다.
- `fused=True`: 파라미터 전체를 한 커널로 업데이트.
- 메모리: fp32 가중치 + m + v + grad ≈ 16 bytes/param → 50M × 16 = 0.8GB. 50M 에서 optimizer 메모리는 문제가 아니고 **활성값과 logits** 가 지배한다(§7.5).

### 7.3 LR 스케줄

- **Warmup 100 step**: Adam 의 m, v 추정이 처음엔 노이즈라서 LR 을 0 부터 선형 증가.
- **Cosine decay → 10%**: 큰 LR 로 넓게 탐색 후 작은 LR 로 수렴. 끝을 0 이 아니라 10% 에 두는 것이 loss 가 조금 더 낮다는 보고(Chinchilla).
- LR 1.5e-3: 50M 급 모델은 1e-3 ~ 3e-3 이 보통 최적(SmolLM-135M 은 3e-3). QK-Norm 이 있어 높은 쪽도 안전. **sweep 대상: {6e-4, 1e-3, 1.5e-3, 3e-3}.**
- `scheduler: wsd`(Warmup-Stable-Decay)도 구현돼 있다. 일정 LR 로 가다가 마지막 20% 만 감쇠한다. "20B 토큰 후 loss 를 보고 40B 로 연장" 하려면 cosine 은 처음부터 총 step 을 알아야 하지만 WSD 는 중간 체크포인트에서 decay 만 다시 돌리면 된다. 1B 모델에서 유용.

### 7.4 Gradient Clipping

global norm 을 1.0 으로 자른다. loss spike 때 한 step 이 가중치를 망치는 것을 막는다. 로그의 `gnorm` 이 지속적으로 1.0 을 크게 넘으면 LR 이 너무 높다는 신호다. 초기 몇 step 의 큰 gnorm(실측 7~25)은 정상이다.

### 7.5 Mixed Precision: FP16 vs BF16

| | FP32 | BF16 | FP16 |
|---|---|---|---|
| 지수 비트 | 8 | 8 | 5 |
| 가수 비트 | 23 | 7 | 10 |
| 표현 범위 | ±3e38 | ±3e38 | ±65,504 |
| GPU | 전부 | **Ampere 이상** (A10G ✓, RTX 8000 ✗) | Turing 이상 |

- BF16 은 범위가 fp32 와 같아 overflow/underflow 걱정이 없다. 정밀도는 낮지만 LLM 학습엔 충분하다. Server 1 (A10G) 기본값.
- FP16 은 범위가 좁다. gradient 가 1e-8 수준이면 0 으로 **underflow** 한다. 해결이 **GradScaler**: loss 에 큰 수 S(초기 2^14)를 곱해 backward 하고, optimizer 전에 S 로 나눈다. gradient 에 inf/NaN 이 생기면 그 step 을 **건너뛰고** S 를 절반으로 줄이며, 2000 step 연속 정상이면 S 를 2 배로 키운다. 로그의 `scale` 열이 이 값이다. 계속 줄어들면 어딘가 overflow 가 반복되는 것이다.
- `torch.autocast` 가 matmul 은 fp16, softmax/norm/loss 같은 민감 연산은 fp32 로 자동 분배한다. 가중치 원본(master weights)은 항상 fp32 로 유지한다.
- Server 2 는 RTX 8000(Turing, sm75)이라 bf16 하드웨어가 없다. `precision: auto` 는 `torch.cuda.is_bf16_supported()` 로 자동 판별한다. **QK-Norm, fp32 RMSNorm/RoPE, fp32 loss 가 모두 fp16 을 위한 방어선**이다.

메모리 실측(50M, seq 2048): micro_batch 8 → 14.1GB, micro_batch 16 → 27.2GB, micro_batch 16 + `torch.compile` → 14.5GB (compile 이 중간 텐서를 융합해 절반으로 줄인다). logits `16 × 2048 × 32768` 가 fp16 2GB + fp32 4GB 로 가장 큰 단일 텐서다. 어휘가 클수록 이 비용이 커진다는 점도 32K 선택의 이유다.

### 7.6 Gradient Accumulation 과 DDP

- 목표 배치 = 524,288 tokens/step. 작은 모델엔 0.5M 토큰이 일반적(Chinchilla 소형 모델, SmolLM). 더 크면 step 수가 줄어 토큰 효율 하락, 더 작으면 gradient 노이즈 증가.
- 한 GPU 에 16 seq 를 올리고 8 번 forward/backward 를 누적(`grad_accum_steps`)한 뒤 한 번 step 한다. 수학적으로는 배치 128 과 동일(loss 를 8 로 나눠 평균). 메모리가 모자라면 micro_batch 를 절반으로, accum 을 2 배로 바꾸면 결과는 같다.
- **DDP**: 두 GPU 가 같은 모델 복제본을 갖고 다른 데이터를 처리한 뒤 gradient 를 `all_reduce` 로 평균한다. NVLink(NV2)로 연결돼 있어 50M 의 gradient 100MB 동기화는 1ms 수준이다. accumulation 중에는 `require_backward_grad_sync=False` 로 통신을 끄고 마지막 micro step 에서만 동기화한다.
- **대안.** FSDP(가중치 샤딩): 모델이 GPU 메모리에 안 들어갈 때 필요. 1B 모델 + optimizer 상태 ≈ 16GB 로 48GB 에 들어가므로 1B 까지도 DDP 로 가능하다. FSDP 는 학습 목적으로 1B 단계에서 실험한다.

### 7.7 Checkpoint

디렉터리 단위로 저장한다: `model.pt`(DDP/compile 래퍼를 벗긴 state_dict), `optimizer.pt`, `trainer_state.pt`(step, tokens_seen, GradScaler, **데이터로더 커서**, RNG), `model_config.yaml`, `train_config.yaml`. 임시 디렉터리에 쓴 뒤 `rename` 하므로 저장 도중 죽어도 반쪽 체크포인트가 남지 않는다. `latest` 는 심볼릭 링크가 아닌 텍스트 파일이라 어떤 파일시스템에서도 동작한다. `--resume latest` 로 재시작하면 데이터 순서까지 그대로 이어진다(실측: step 60 → 70 재시작 확인).

---

## 8. 생성 (Inference)

### 8.1 KV Cache

토큰을 하나 생성할 때마다 전체 시퀀스를 다시 계산하면 O(T²). 이미 계산한 과거 토큰의 K, V 를 층마다 저장해 두면 새 토큰 하나의 q 만 계산하고 과거 K, V 에 attention 하면 된다(O(T)). `KVCache` 클래스가 층별로 K, V 를 이어붙이고, `start_pos` 로 RoPE 위치를 맞춘다. 테스트 `test_kv_cache_matches_full_forward` 가 "캐시 사용 결과 == 전체 재계산 결과"를 1e-4 오차로 검증한다. 캐시 사용 중에는 q 블록이 k 블록의 **꼬리** 이므로 정사각 causal mask 가 아니라 `tril(diagonal = T_k − T)` 마스크를 쓴다.

### 8.2 Sampling

- **temperature** τ: logits / τ. τ→0 은 argmax(greedy, 반복 많음), τ>1 은 무작위.
- **top-k**: 확률 상위 k 개만 남김.
- **top-p (nucleus)**: 누적 확률 p 까지만 남김. 분포가 뾰족하면 적게, 평평하면 많이 남기므로 top-k 보다 적응적.
- **repetition penalty**: 이미 나온 토큰의 logit 을 깎는다. 작은 모델의 "같은 말 반복"을 완화.

---

## 9. Ollama 의 오픈 모델과 무엇이 다른가

Ollama 로 받는 `llama3.2`, `qwen3`, `gemma3`, `mistral`, `phi4` 등과 ResearchLM-50M 의 차이를 정리한다. 결론부터: **아키텍처는 거의 같은 설계 철학이고, 차이는 규모·데이터·후처리·배포 형식에 있다.**

### 9.1 아키텍처 비교

| 구성 요소 | ResearchLM-50M | Llama 3.2 1B | Qwen3 0.6B | Gemma 3 1B | Mistral 7B |
|---|---|---|---|---|---|
| 구조 | Decoder-only | Decoder-only | Decoder-only | Decoder-only | Decoder-only |
| Norm | RMSNorm pre | RMSNorm pre | RMSNorm pre | RMSNorm pre+post | RMSNorm pre |
| 위치 | RoPE θ=1e4 | RoPE θ=5e5 + scaling | RoPE θ=1e6 | RoPE (local 1e4 / global 1e6) | RoPE θ=1e6 |
| Attention | GQA 8/2 | GQA 32/8 | GQA 16/8 | GQA 4/1, local+global 교차 | GQA 32/8, sliding window |
| QK-Norm | ✓ | ✗ | ✓ | ✓ | ✗ |
| FFN | SwiGLU | SwiGLU | SwiGLU | GeGLU | SwiGLU |
| Bias | 없음 | 없음 | 없음 (q/k/v 도 없음, Qwen2 는 있었음) | 없음 | 없음 |
| Tying | ✓ | ✓ | ✓ | ✓ | ✗ |
| Vocab | 32K | 128K | 152K | 262K | 32K |
| Context | 2K | 128K | 32K | 32K | 32K |
| 파라미터 | 50M | 1.24B | 0.6B | 1B | 7.2B |
| 학습 토큰 | 1B 목표 | ~9T (8B 로부터 distill/prune) | 36T | 2T | 비공개 (~8T 추정) |

읽는 법: RoPE, RMSNorm, GQA, SwiGLU, no-bias, tying 은 **2023 년 이후 소형 모델의 공통 분모**다. ResearchLM 은 그 공통 분모를 그대로 쓰고, 여기에 QK-Norm 을 더한 구성이라 Qwen3·Gemma 3 와 가장 닮았다. 반대로 Llama 3 와 Mistral 은 QK-Norm 이 없는 대신 bf16 과 매우 낮은 LR 로 안정성을 확보한다.

### 9.2 본질적 차이 다섯 가지

1. **규모와 데이터 양.** 50M vs 0.6B~70B, 1B 토큰 vs 수십 T 토큰. 지식과 추론 능력의 거의 전부는 여기서 나온다. 50M 모델은 문법과 간단한 사실 연관은 배우지만 세계 지식·추론은 기대할 수 없다. 1B 모델도 Ollama 소형 모델 대비 학습 토큰이 1/100 ~ 1/1000 수준이다.

2. **Post-training 의 부재.** Ollama 의 기본 태그(`llama3.2`, `qwen3`)는 **Instruct/Chat** 모델이다. 사전학습 뒤에 SFT(지시-응답 데이터 미세조정) → 선호 최적화(DPO/RLHF) → 안전성 튜닝을 거쳤다. ResearchLM-50M 은 **순수 base 모델**이다. "질문하면 답한다"가 아니라 "텍스트를 이어 쓴다". 질문을 넣으면 질문 목록을 계속 생성하는 것이 정상 동작이다. 예약해 둔 `<|im_start|>` 계열 토큰이 나중에 SFT 를 붙이기 위한 자리다.

3. **토크나이저 규모와 다국어.** 128K~262K 어휘는 100 개 이상 언어와 코드를 효율적으로 압축한다. 32K 는 영어+한국어+코드+수학 네 도메인에 집중한 선택이다. 대신 한국어 압축률은 우리가 코퍼스 비율로 직접 제어할 수 있다. Llama 3 는 한국어 데이터 비율이 낮아 한국어 토큰 효율이 영어의 절반 이하다.

4. **컨텍스트 길이.** 2K vs 32K~128K. 긴 컨텍스트는 (a) RoPE θ 를 키우고 (b) 긴 문서로 continued pretraining 하고 (c) sliding-window/local-global attention 으로 비용을 억제해 얻는다. 50M 모델에 128K 컨텍스트는 의미가 없다. 300M 부터 4K 로 늘리는 계획이 goal.md 에 있다.

5. **배포 형식과 양자화.** Ollama 모델은 **GGUF** 파일로, 가중치가 4~8 bit 로 양자화(Q4_K_M 등)되어 llama.cpp 런타임에서 CPU/GPU 로 돈다. ResearchLM 체크포인트는 fp32 PyTorch `state_dict` 다. llama.cpp 의 `convert_hf_to_gguf.py` 는 HF 포맷의 Llama 호환 구조를 요구하므로, 나중에 Ollama 에서 돌리려면 (a) `ResearchLM` → HF `LlamaForCausalLM` 포맷 변환기(가중치 이름 매핑, QK-Norm 은 Qwen3 포맷이 호환), (b) SentencePiece → HF tokenizer 변환, (c) GGUF 변환 + Modelfile 작성이 필요하다. 구조를 Llama/Qwen3 와 호환되게 설계한 덕분에 변환 자체는 가능하다.

### 9.3 "그럼 왜 직접 만드나"

Ollama 모델은 결과물이다. ResearchLM 은 **그 결과물이 어떻게 만들어지는지를 통제 가능한 규모에서 전부 재현**하는 것이 목표다. 데이터 비율을 바꾸면 한국어 성능이 어떻게 변하는지, QK-Norm 을 빼면 어느 LR 에서 발산하는지, GQA 그룹 수가 loss 에 얼마나 영향을 주는지는 공개 모델을 다운로드해서는 알 수 없다.

---

## 10. 학습 실패 진단 체크리스트

| 증상 | 가장 흔한 원인 | 확인 / 조치 |
|---|---|---|
| 초기 loss 가 10.4 가 아님 | 어휘 크기 불일치, 초기화 버그, target 이 input 과 같음 | `scripts/model_info.py`, `tests/test_model.py` |
| loss 가 10.4 에서 안 내려감 | LR 0 (warmup 설정), 데이터가 전부 EOS/pad, gradient 가 0 | 로그의 `lr`, `gnorm` 확인. gnorm 0 이면 backward 미연결 |
| loss 가 1~2 step 만에 NaN | fp16 overflow(QK-Norm off 시), LR 너무 큼 | `scale` 이 급감하는지, QK-Norm on, LR 절반 |
| loss 가 수백 step 뒤 급등(spike) 후 복구 안 됨 | LR 너무 큼, β2 0.999, 데이터 오염(초장문 반복) | LR 절반, β2 0.95 확인, spike 직전 배치 덤프 |
| `scale` 이 계속 반으로 줄어 1 미만 | 어떤 연산이 매 step inf 를 냄 | RMSNorm/RoPE 가 fp32 인지, 어휘 밖 id (`>= vocab_size`) 인지 |
| train loss ↓ val loss ↑ | 과적합(epoch 수 과다) | `describe()` 출력의 "epochs over the run" 확인. 샘플 코퍼스는 3 epoch 이내 |
| val loss 가 train 보다 훨씬 낮음 | val 에 train 문서가 섞임(dedup 실패) | `tokenize_data.py` 의 split, 추후 near-dedup |
| 생성이 같은 구절 반복 | 작은 모델의 일반적 현상 + greedy | temperature 0.7~0.9, top-p 0.9, repetition penalty 1.1 |
| 한국어만 품질 나쁨 | 토크나이저 한국어 비율 부족, 데이터 비율 | `eval_tokenizer.py` bytes/token, 한국어 비율 ablation |
| 두 GPU 가 다른 loss 를 찍음 | DDP 동기화 실패 | `all_reduce_mean` 로그는 평균값. rank 별 출력은 `is_main` 외에 없음 → `check_ddp.py` 로 통신 확인 |
| OOM | logits(fp32 2GB) + 활성값 | micro_batch 절반, grad_accum 2 배 (수학적으로 동일) |
| tok/s 가 낮음 | Turing 은 FlashAttention 미지원, 작은 모델은 커널 launch 지배 | `compile: true` (실측 1.65 배), micro_batch 를 메모리 한계까지 키움. compile 첫 step 은 40 초 정도 느린 것이 정상 |

---

## 11. 실행 명령어 (Server 2 기준)

모든 명령은 `conda activate researchlm` 후 저장소 루트에서 실행한다. 경로는 `/storage/researchlm/` 기준이다.

### 11.1 단위 테스트

```bash
python tests/test_model.py
```

### 11.2 토크나이저 학습 (v0: 현재 샘플 / v1: 정제 코퍼스)

```bash
python scripts/train_tokenizer.py \
  --input english=data_samples/english.jsonl \
  --input korean=data_samples/korean.jsonl \
  --input math=data_samples/math.jsonl \
  --weights english=0.5,korean=0.25,math=0.10 \
  --out-prefix /storage/researchlm/tokenizer/researchlm_32k

python scripts/eval_tokenizer.py \
  --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
  --input english=data_samples/english.jsonl \
  --input korean=data_samples/korean.jsonl \
  --input math=data_samples/math.jsonl --limit 2000
```

정제 코퍼스가 준비되면 `--input` 에 `/storage/researchlm/cleaned/<domain>.jsonl` 과 `code=...` 를 추가하고 `--weights english=0.5,korean=0.25,code=0.15,math=0.10` 으로 다시 학습한다.

### 11.3 토큰 샤드 생성

```bash
python scripts/tokenize_data.py \
  --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
  --input english=data_samples/english.jsonl \
  --input korean=data_samples/korean.jsonl \
  --input math=data_samples/math.jsonl \
  --out-dir /storage/researchlm/tokenized/samples_v0 \
  --val-fraction 0.02 --workers 24
```

### 11.4 Tiny 디버그 (1 GPU, 약 1 분)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --model configs/model/tiny_debug.yaml \
  --train configs/train/tiny_debug.yaml
```

### 11.5 50M pilot (2 GPU, 샘플 코퍼스, 약 12 분)

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --model configs/model/researchlm_50m.yaml \
  --train configs/train/50m_pilot_samples.yaml
```

### 11.6 50M 정식 run (2 GPU, 1B 토큰 ≈ 2 시간, 정제 코퍼스 준비 후)

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --model configs/model/researchlm_50m.yaml \
  --train configs/train/50m_server2.yaml
```

### 11.7 재시작 / 하이퍼파라미터 오버라이드

```bash
# 마지막 체크포인트에서 이어서
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --model configs/model/researchlm_50m.yaml \
  --train configs/train/50m_server2.yaml --resume latest

# LR sweep 예: run_name 을 바꿔야 체크포인트가 섞이지 않는다
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --model configs/model/researchlm_50m.yaml \
  --train configs/train/50m_pilot_samples.yaml \
  --set run_name=50m-lr3e-3 --set lr=3e-3
```

### 11.8 생성

```bash
python scripts/generate.py \
  --ckpt /storage/researchlm/checkpoints/researchlm-50m-pilot \
  --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
  --prompt "대한민국의 수도는" --prompt "The derivative of x^2 is" \
  --max-new-tokens 100 --temperature 0.8 --top-p 0.95
```

### 11.9 로그 보기

```bash
tail -f logs/researchlm-50m-pilot.jsonl          # step 별 JSON 한 줄
python - <<'PY'
import json; rows=[json.loads(l) for l in open("logs/researchlm-50m-pilot.jsonl")]
for r in rows:
    if "val_loss" in r: print(r["step"], round(r["val_loss"],4))
PY
```

### 11.10 Server 1 (A10G, 1 GPU, BF16) 에서 같은 코드 실행

```bash
python scripts/train.py --model configs/model/researchlm_50m.yaml \
  --train configs/train/50m_pilot_samples.yaml --server configs/server/server1.yaml \
  --set grad_accum_steps=16          # GPU 1 개이므로 accum 2 배로 같은 배치 유지 (A10G 24GB: micro_batch 16 + compile 은 ~15GB)
```

---

## 12. 파일 맵

```text
researchlm/
├── model/
│   ├── config.py        ModelConfig: 모든 shape 하이퍼파라미터 + 해석적 파라미터 수
│   ├── norm.py          RMSNorm (fp32 통계)
│   ├── rope.py          RoPE 테이블 생성 / 적용 (fp32 회전)
│   ├── attention.py     GQA + QK-Norm + SDPA + KVCache
│   ├── mlp.py           SwiGLU
│   ├── block.py         pre-norm 블록
│   └── transformer.py   ResearchLM: 임베딩, 블록 스택, tying, 초기화, loss, generate
├── data/
│   ├── tokenizer.py     SentencePiece 학습 설정 + Tokenizer 래퍼
│   ├── shards.py        uint16 샤드 쓰기/읽기(memmap)
│   └── dataset.py       도메인 믹싱 + 재시작 가능한 윈도우 샘플러
├── training/
│   ├── config.py        TrainConfig (yaml + server yaml + --set 오버라이드)
│   ├── optimizer.py     AdamW, decay / no-decay 그룹
│   ├── scheduler.py     warmup + cosine / wsd / linear
│   ├── checkpoint.py    원자적 저장, resume, 모델만 로드
│   ├── distributed.py   torchrun 환경 변수 → DDP 초기화
│   └── trainer.py       학습 루프, autocast, GradScaler, 로그, eval
└── eval/
    └── tokenizer_metrics.py  chars/token, bytes/token
scripts/
├── train_tokenizer.py   코퍼스 샘플링 + SentencePiece 학습
├── eval_tokenizer.py    압축률 표 + 토큰 분해 예시
├── tokenize_data.py     JSONL → 샤드 (multiprocessing)
├── train.py             학습 진입점 (단일 GPU / torchrun)
├── generate.py          체크포인트에서 샘플링
├── model_info.py        파라미터·FLOPs·메모리 추정
└── check_ddp.py         NCCL all_reduce 스모크 테스트
configs/
├── model/researchlm_50m.yaml, tiny_debug.yaml
├── train/tiny_debug.yaml, 50m_pilot_samples.yaml, 50m_server2.yaml
├── data/mixture_v1.yaml
└── server/server1.yaml, server2.yaml
tests/test_model.py      파라미터 수, causality, KV cache 일치, RoPE 상대성, 스케줄 등 11 개
```

---

## 13. 용어 사전

- **Token**: 토크나이저가 만드는 최소 단위. 단어보다 작고 글자보다 큰 경우가 많다.
- **Embedding**: 토큰 id 를 d_model 차원 실수 벡터로 바꾸는 룩업 테이블.
- **Residual stream**: 블록을 통과하며 계속 더해지는 `x`. 모든 층이 읽고 쓰는 공용 메모리.
- **Logits**: softmax 전의 점수. `lm_head` 출력, shape `[T, vocab]`.
- **Perplexity (ppl)**: `exp(loss)`. "모델이 평균적으로 몇 개 후보 사이에서 헤매는가".
- **Causal mask**: 미래 토큰을 보지 못하게 하는 마스크. decoder-only 의 정의.
- **Head**: attention 을 여러 개의 작은 부분공간(64 차원)에서 병렬로 수행하는 단위.
- **GQA / MQA / MHA**: KV head 를 그룹 공유 / 하나만 / head 마다 두는 방식.
- **KV cache**: 생성 시 과거 토큰의 K, V 를 저장해 재계산을 피하는 메모리.
- **Warmup**: 학습 초반 LR 을 0 부터 올리는 구간.
- **Weight decay**: 가중치를 매 step 조금씩 0 으로 끌어당기는 정규화.
- **Gradient accumulation**: 작은 micro-batch 여러 번의 gradient 를 더해 큰 배치를 흉내 내는 기법.
- **DDP**: Distributed Data Parallel. GPU 마다 모델 복제, 데이터 분할, gradient 평균.
- **FSDP**: Fully Sharded Data Parallel. 가중치·gradient·optimizer 상태까지 GPU 간 분할.
- **MFU**: Model FLOPs Utilization. 이론 peak 대비 실제 유용한 연산 비율.
- **Chinchilla-optimal**: 주어진 연산량에서 loss 를 최소화하는 N, D 조합 (D ≈ 20N).
- **SFT / RLHF / DPO**: 사전학습 뒤 지시를 따르게 만드는 후처리 단계들.
- **GGUF**: llama.cpp / Ollama 가 쓰는 양자화 가중치 파일 포맷.

---

## 14. 참고 문헌 (설계 근거)

- Vaswani et al. 2017, *Attention Is All You Need* — Transformer, scaled dot-product attention
- Radford et al. 2019, *GPT-2* — decoder-only, pre-norm, residual 초기화 축소
- Hoffmann et al. 2022, *Training Compute-Optimal LLMs (Chinchilla)* — 20 tokens/param
- Su et al. 2021, *RoFormer* — RoPE
- Zhang & Sennrich 2019, *RMSNorm*
- Shazeer 2020, *GLU Variants Improve Transformer* — SwiGLU
- Ainslie et al. 2023, *GQA*
- Dehghani et al. 2023 (ViT-22B), Wortsman et al. 2023 — QK-Norm 과 소형 모델 불안정성
- Touvron et al. 2023, *Llama 2*; Dubey et al. 2024, *Llama 3* — 전체 레시피 참고
- Kudo & Richardson 2018, *SentencePiece*
- Dao 2023, *FlashAttention-2* — SDPA 커널 배경
