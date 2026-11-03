# ResearchLM 데이터 파이프라인과 1B 스케일링 가이드

> [researchlm_50m_guide.md](researchlm_50m_guide.md) 가 "모델이 무엇인가"를 다뤘다면, 이 문서는
> **"1B 모델을 학습시킬 20B 토큰을 어떻게 만들고, 50M → 150M → 300M → 1B 로 어떻게 올라가는가"** 를 다룬다.
> goal.md 의 Step 1 ~ Step 11 과 Phase 5 ~ 8 에 대응한다. 서술 순서는 같다.
> **무엇인가 → 왜 그렇게 하는가 → 다른 방법은 무엇인가 → 왜 지금은 덜 맞는가.**

---

## 0. 한 장 요약

```text
HF parquet ──download_data.py──▶ raw/<domain>/*.parquet          (영어·한국어·수학, 메타데이터 포함)
Stack-Edu 메타 ──fetch_stack_edu.py (SWH S3)──▶ raw/code/<Lang>/*.jsonl.zst
        │
        ▼ data_structure_analyzer.py  → metadata/analysis/*.png, *_summary.json, *_examples.jsonl
        ▼ filter_data.py  (filters_v1.yaml, 도메인별 규칙) → cleaned/<domain>/   + rejected_samples.txt
        ▼ dedup_data.py   (sha1 exact + MinHash/LSH near)   → cleaned/<domain>_dedup/
        ▼ decontaminate_data.py (벤치마크 13-gram)          → cleaned/<domain>_final/
        ▼ train_tokenizer.py (v1, 정제 코퍼스로 재학습·동결)  → tokenizer/researchlm_32k.model
        ▼ tokenize_data.py                                    → tokenized/v1/<domain>/train_*.bin
        ▼ train.py  50M → 150M → 300M → 1B
```

| 단계 | 상태 (2026-10-26) | 검증 규모 |
|---|---|---|
| 다운로드 | 코드 완성. 시험 파일은 임시 코드로 받았고 `download_data.py` 전체 실행은 이번 run 이 처음 | 104,680 문서 |
| Stack-Edu 수집 | 코드 완성, 209 files/s 실측, resume 확인 | 3,500 파일 |
| 구조 분석기 | 코드 완성, 수학·코드 실행, 그래프 32 장 | 25,000 문서 |
| 필터 v1 | 코드 완성, 수학 83.8% / 코드 98.2% 유지 | 108,180 문서 |
| Dedup | 코드 완성, 수학 실행: exact 0%, near 0.86% | 87,725 문서 |
| Decontamination | 벤치마크 8 종 69,645 항목 → 13-gram 347 만 개, 수학 0.99% 제거 (AoPS/AMC 페이지 등 실제 오염) | 86,973 문서 |
| Tokenizer v1 | 스크립트가 정제 출력(zst/dir) 을 읽도록 확장, 정제 수학 → 126M 토큰 샤드 생성 확인 | 84,119 문서 |
| 150M / 300M / 1B config | 파라미터 수 151M / 304M / 997M 확인, 1B 2-GPU 메모리 실측 29.6GB | |

실제 20B 토큰 규모 실행은 사용자가 11 절의 명령 순서대로 돌린다.

---

## 1. 디렉터리 레이아웃 (`/storage/researchlm`)

```text
raw/
  english/   fineweb-edu-dedup/train-000NN-of-00234.parquet   manifest.json
  korean/    data/kor_Hang/train/000_000NN.parquet            manifest.json
  math/      finemath-4plus/train-000NN-of-00064.parquet      manifest.json
  code_meta/ <Language>/train-*.parquet                       (Stack-Edu 메타데이터)
  code/      <Language>/code_NNNNN.jsonl.zst                  (SWH 에서 받은 본문)
cleaned/
  <domain>/         filtered_fNNNN_NNNNN.jsonl.zst  filter_stats.json  rejected_samples.txt
  <domain>_dedup/   dedup_fNNNN_NNNNN.jsonl.zst     dedup_stats.json
  <domain>_final/   final_fNNNN_NNNNN.jsonl.zst     decontam_stats.json  contaminated_samples.txt
metadata/
  analysis/         <domain>_features.parquet  <domain>_summary.json  <domain>_*.png  <domain>_examples.jsonl
  benchmarks/       mmlu.jsonl kmmlu.jsonl gsm8k.jsonl math.jsonl humaneval.jsonl mbpp.jsonl arc.jsonl hellaswag.jsonl
                    ngram13.npy  ngram13.json
tokenizer/          researchlm_32k.{model,vocab,json}  researchlm_32k_eval.json
tokenized/v1/       <domain>/train_NNNNN.bin  val_NNNNN.bin  manifest.json
checkpoints/        <run_name>/step_NNNNNN/
```

규칙 하나를 지켜야 한다. **데이터 디렉터리 안의 `.jsonl` / `.jsonl.zst` / `.parquet` 는 전부 문서로 취급된다.** 그래서 통계·검토용 파일은 `.json`, `.txt`, `.log` 확장자를 쓴다. 이 규칙을 어기면 다음 단계가 거부 샘플을 학습 데이터로 읽는다 (실제로 개발 중 한 번 발생해 고쳤다).

---

## 2. Step 1 — 원본 다운로드 (`scripts/download_data.py`)

**무엇.** Hugging Face 의 parquet 파일을 통째로 받아 `raw/` 에 둔다. 토큰 예산에 도달할 때까지 파일을 순서대로 받는다.

**왜 parquet 통째로.** `datasets` 스트리밍은 재시작이 불가능에 가깝고(`skip(n)` 이 처음부터 다시 읽음), 메타데이터를 매번 파싱한다. parquet 파일 단위 다운로드는 `hf_hub_download` 가 이어받기를 지원하고, 컬럼 단위로 읽을 수 있으며(`text` 만 읽으면 I/O 1/3), row group 단위로 병렬 처리가 된다.

**토큰 집계.** FineWeb-Edu 의 `token_count` 는 `metadata` 구조체 안에 있어 파일별 합산 시 구조체를 읽는다.

**토큰 예산 1.3 배.** 필터와 dedup 으로 15~30% 가 빠진다. 목표 20B 의 1.3 배인 26B 를 받는다.

| 도메인 | 저장소 | 파일 | 파일당 | 필요 파일 | 다운로드 |
|---|---|---:|---:|---:|---:|
| English | smollm-corpus / fineweb-edu-dedup | 234 × 2.2 GiB | ≈ 0.94B tok | 14 | ≈ 31 GiB |
| Korean | fineweb-2 / kor_Hang | 25 × 4.0 GiB | ≈ 1.2B tok (추정) | 6 | ≈ 27 GiB |
| Math | finemath / finemath-4plus | 64 × 274 MiB | ≈ 0.15B tok | 17 | ≈ 4.6 GiB |
| Code meta | stack-edu / 10 개 언어 | 32 파일 | 메타만 | 전부 | ≈ 12 GiB |

**토큰 수 세기.** FineWeb-Edu, FineMath 는 `token_count` 컬럼(GPT-2 토크나이저 기준)이 있어 합산한다. FineWeb2 Korean 은 없으므로 문자 수 ÷ 2.1 (v0 토크나이저 실측 chars/token)로 추정한다. 우리 토크나이저와 10~20% 차이가 나므로 1.3 배 여유가 그 오차도 흡수한다.

**대안.** 전체 데이터셋 다운로드(512 GiB 영어): 필요 없음. `datasets.load_dataset(streaming=True)`: 재시작 불가. HF `snapshot_download` + `allow_patterns`: 가능하지만 토큰 예산 기반 중단을 못 한다.

---

## 3. Step 2 — Stack-Edu 코드 수집 (`scripts/fetch_stack_edu.py`)

**문제.** HF 의 Stack-Edu 는 `blob_id`(Software Heritage 의 sha1), `repo_name`, `path`, `score`, `license_type` 만 있다. 본문은 **Software Heritage 아카이브**에 있다.

**어떻게.** 공개 S3 버킷 `s3://softwareheritage/content/<blob_id>` 에 gzip 으로 저장돼 있고 익명 접근이 된다(`boto3` + `UNSIGNED`). 요청당 약 0.25 초지만 64 스레드로 병렬화하면 **209 files/s** 가 나온다. 3.9B 토큰 ≈ 230 만 파일 ≈ 3 시간.

**선택 규칙** (`configs/data/download_v1.yaml` 의 `stack_edu`):

- `int_score >= 3`: Stack-Edu 의 교육적 가치 점수(0~5). 3 이상이 StarCoder2 가 쓴 기준.
- `license_type == permissive`: MIT, Apache, BSD 계열만. 라이선스 분쟁 회피.
- 200 B ≤ 크기 ≤ 200 KB: 너무 작으면 빈 파일, 너무 크면 데이터/생성 코드.
- 언어 비율: Python 30%, Java 12%, JavaScript 12%, C++ 10%, C 7%, Rust 7%, Go 6%, TypeScript 6%, SQL 5%, Shell 5%. Python 을 크게 둔 이유는 HumanEval / MBPP 평가가 Python 이기 때문이다.
- 시드 고정 셔플 후 예산까지 선택 → 파일이 저장소 순서로 편중되지 않는다.

**Resume.** 출력 디렉터리의 `blob_id` 를 읽어 이미 받은 것은 건너뛴다. 중간에 죽어도 같은 명령을 다시 치면 된다.

**대안.** SWH REST API: 시간당 1,200 요청 제한으로 불가. `common-pile/stackv2_edu`: 본문이 포함된 유사 데이터셋이라 지름길이 되지만, goal.md 의 "Stack-Edu 를 직접 복원한다"는 학습 목표에서 벗어난다. 문제가 생기면 대체 후보.

---

## 4. Step 3 — 구조 분석기 (`scripts/data_structure_analyzer.py`)

**무엇.** 도메인당 2 만 문서를 reservoir sampling 으로 뽑아 `researchlm/data/quality.py` 의 특징 40 여 개와 메타데이터를 한 표(parquet)로 만들고, 백분위 요약·히스토그램·"메타 점수 vs 휴리스틱" 산점도를 그린다.

**왜 reservoir sampling.** 파일 앞부분만 보면 크롤 시점·도메인이 편향된다. 한 번의 스트리밍으로 균등 샘플을 얻는 알고리즘이다.

**산출물 읽는 법 (수학 샘플 결과).**

| 특징 | p10 | p50 | p90 | p99 | 해석 |
|---|---:|---:|---:|---:|---:|
| n_chars | 877 | 2,897 | 9,876 | 36,910 | 길이 하한 200 은 0.3% 만 자름 |
| meta_score | 3.56 | 3.91 | 4.47 | 4.84 | FineMath-4+ 는 이미 3.5 이상 |
| meta_language_score | 0.64 (p10) | 0.88 | | | p05 가 0.50 → 임계 0.60 은 5.7% 제거. 수식이 많으면 언어 점수가 낮아지므로 **검토 대상** |
| alnum_ratio | 0.62 | 0.74 | | | p05 0.58, p01 0.49 → 임계 0.45 |
| dup_10gram_frac | 0 | 0.05 | | | 평균 0.26: 반복이 긴 꼬리를 가짐 |
| dup_line_frac | 0 | 0.014 | 0.26 | 0.50 | 임계 0.50 = p99 |

`score_vs_*.png` 산점도는 "데이터셋이 매긴 품질 점수와 우리 휴리스틱이 얼마나 일치하는가"를 보여준다. 상관이 약하면 두 신호가 서로 다른 것을 잡고 있으니 둘 다 쓰는 것이 맞고, 강하면 하나로 충분하다. `*_summary.json` 의 `corr_with_meta_score` 가 수치다.

---

## 5. Step 4 — 수동 검토

필터가 "정상 데이터를 버리지 않도록" 임계값을 정하는 유일한 방법은 **직접 읽는 것**이다. 도구는 두 가지다.

1. `metadata/analysis/<domain>_examples.jsonl`: 메타 점수 기준 하위 5 / 중간 5 / 상위 5 문서.
2. `cleaned/<domain>/rejected_samples.txt`: 거부 사유별 10 개 예시.

절차: 거부 사유 상위 5 개에 대해 예시 10 개씩 읽고, "이건 버리면 안 된다"가 3 개 이상이면 그 규칙의 임계값을 완화한다. `filters_v1.yaml` 의 `history` 에 날짜·변경·이유를 적는다. 바뀐 설정은 `version` 을 올린다 (`v1-draft` → `v1`).

---

## 6. Step 5 — Data Filter v1 (`researchlm/data/filters.py`, `configs/data/filters_v1.yaml`)

**구조.** 규칙 = (특징, 비교, 임계값, 이름). 위에서 아래로 평가하고, **처음 실패한 규칙이 거부 사유**가 된다. 그래서 통계를 보면 어느 규칙이 데이터를 얼마나 잘라내는지 바로 보인다. 메타데이터 규칙(`meta.*`)은 컬럼이 없으면 건너뛴다.

**도메인별 규칙 요약과 근거.**

| 규칙 | English | Korean | Math | Code | 근거 |
|---|---|---|---|---|---|
| 최소 길이 | 200 자 / 50 단어 | **100 자** | 200 자 / 40 단어 | 200 자 / 5 줄 | 한국어 중앙값 694 자, 영어 2,740 자 (goal.md §11) |
| 문자 구성 | latin ≥ 0.60, alnum ≥ 0.70 | **hangul ≥ 0.35**, latin ≤ 0.45, cjk ≤ 0.10, alnum ≥ 0.55 | alnum ≥ 0.45 | alpha ≥ 0.25, digit ≤ 0.35 | 한영 혼합 기술문서 허용, 한자·일본어 페이지 제거 |
| 문법 신호 | stopword ≥ 0.08 | **조사/어미 ≥ 1.0 / 100자** | | | 메뉴·키워드 나열은 조사가 없음 |
| 단어 길이 | 3 ≤ mean ≤ 10 | | | | Gopher |
| URL / HTML | url ≤ 10%, html ≤ 5 | 같음 | url ≤ 10%, html ≤ 10 | | |
| 반복 문자 | ≤ 0.05 | ≤ 0.08 | **≤ 0.20** | | 수학 구분선 `----` |
| 중복 줄 | ≤ 0.30 | ≤ 0.30 | **≤ 0.50** | | "Solution" 헤더 반복 |
| top 2/3-gram | ≤ 0.20 / 0.18 | ≤ 0.25 | | | Gopher |
| dup 5/10-gram | ≤ 0.30 / 0.20 | 같음 | **10-gram ≤ 0.50** | | 아래 "교훈" 참고 |
| 줄 길이 | | | | mean ≤ 120, max ≤ 1500 | minified JS, 데이터 파일 |
| 바이너리성 | | | | hex/base64 blob ≤ 3 | |
| 메타 | language_score ≥ 0.65, int_score ≥ 2 | language_score ≥ 0.60, minhash_cluster_size ≤ 50 | language_score ≥ 0.60, int_score ≥ 4 | int_score ≥ 3, permissive | 데이터셋 공식 신호 활용 (goal.md §12-4) |

**실측 (v1-draft).**

| 도메인 | 문서 | 유지 | 상위 거부 사유 |
|---|---:|---:|---|
| Math (FineMath 파일 1 개) | 104,680 | 83.8% (문자 83.1%) | language_score 5.7%, dup_10gram 4.2%, alnum 3.2% |
| Code (Rust 3,500) | 3,500 | 98.2% | alnum 0.7%, autogenerated 0.5%, hex blob 0.4% |

**교훈: n-gram 중복 지표의 정의.** 처음 구현은 중복된 10-gram 마다 글자 수를 전부 더해서, 20 단어짜리 문장이 두 번 나오면 11 개의 10-gram 이 각각 계산돼 실제보다 10 배 과대평가됐다. 그 상태로는 수학 문서 24% 가 잘렸다. Gopher 의 정의대로 **"중복 n-gram 에 속한 단어를 한 번만 세는"** 방식으로 고치고, 같은 시점에 수학 임계값도 0.35 에서 0.50 으로 완화해 4.2% 가 됐다. 두 변경이 동시에 들어갔으므로 감소분을 지표 수정만의 효과로 볼 수 없다. 지표를 쓰기 전에 손으로 만든 예문으로 값이 직관과 맞는지 확인하는 것이 필수다 (`tests/test_data_pipeline.py`).

**대안.** 모델 기반 품질 분류기(FineWeb-Edu 의 `score` 가 그것): 우리는 이미 그 점수를 메타 규칙으로 쓴다. 자체 분류기 학습은 300M 이후 데이터 품질 실험 후보. 퍼플렉시티 필터(KenLM): 위키 스타일에 편향, 코드·수학에 부적합.

---

## 7. Step 6 — Deduplication (`researchlm/data/dedup.py`, `scripts/dedup_data.py`)

**왜.** 중복 문서는 (a) 모델이 외우게 만들고 (b) 토큰 예산을 낭비하며 (c) 같은 문서가 train/val 에 들어가 val loss 를 속인다. Lee et al. 2022 (*Deduplicating Training Data Makes LMs Better*) 가 근거다.

**Exact.** 소문자화·문장부호 제거·공백 정규화 후 sha1. 비용 거의 없음.

**Near (MinHash).**

1. 문서를 단어 5-gram 집합(shingle)으로 본다.
2. 두 집합의 유사도는 Jaccard `J = |A∩B| / |A∪B|`.
3. 해시 함수 h 에 대해 `min(h(A)) == min(h(B))` 일 확률이 정확히 J 다. 해시 128 개의 최소값(서명)을 비교하면 J 를 추정할 수 있다 (`jaccard_estimate`).
4. 모든 쌍을 비교하면 N² 이므로 **LSH**: 서명 128 개를 16 밴드 × 8 행으로 나누고, 어느 한 밴드가 완전히 같으면 후보 쌍으로 본다. 한 밴드가 같을 확률은 `J^8`, 16 밴드 중 하나라도 같을 확률은 `1 − (1 − J^8)^16`. J = 0.7 에서 약 0.61, J = 0.9 에서 0.9999, J = 0.3 에서 0.001 이다. 즉 **0.7 근처가 문턱**이다.
5. 후보 쌍을 union-find 로 묶고 클러스터마다 하나(가장 앞 문서)만 남긴다.

**구현 선택.** 밴드별로 64-bit 해시를 만들어 **정렬**한 뒤 같은 값의 구간을 묶는다. 파이썬 dict 로 버킷을 만들면 2,500 만 문서 × 16 밴드에 수십 GB 가 들지만, 정렬 방식은 문서당 8 바이트 × 16 이다. 서명 저장은 2,500 만 × 128 × 4 B = 12.8 GB.

**비용 추정 (20B 토큰 ≈ 2,500 만 문서).** 서명 0.7 ms/문서 → 64 코어로 5 분. LSH 정렬 16 회 → 수 분. 전체 30 분 이내.

**범위.** 도메인 내부만. 영어와 한국어 사이의 중복은 없고, 수학 ↔ 영어 교차 중복은 FineMath 가 FineWeb 과 다른 파이프라인이라 적다. 교차 dedup 은 150M 단계 실험 후보.

**대안.** `datasketch`: 같은 알고리즘, 메모리 비효율. Suffix array 기반 부분 문자열 중복 제거(Lee et al.): 더 강력하지만 구현·메모리 비용이 크고 20B 규모에선 MinHash 로 충분. 임베딩 기반 semantic dedup: 의미는 같고 표현이 다른 문서를 잡지만 GPU 비용이 크다.

---

## 8. Step 7 — Benchmark Decontamination (`researchlm/data/decontaminate.py`)

**왜.** 평가 세트의 문제가 학습 데이터에 섞여 있으면 벤치마크 점수가 능력이 아닌 암기를 측정한다. 웹 크롤에는 GSM8K 문제를 풀어주는 블로그, MMLU 문항이 그대로 올라온 사이트가 흔하다.

**방법.** 벤치마크의 문항+선택지+정답을 정규화하고 단어 **13-gram** 해시 집합을 만든다 (GPT-3, Llama 2 방식). 학습 문서에 그 13-gram 이 하나라도 있으면 제거한다. 13 단어가 안 되는 짧은 문항은 전체를 하나의 gram 으로 넣는다.

**벤치마크 (현재 인덱스).** MMLU 14,042 / KMMLU 35,030 / GSM8K 1,319 / MATH 5,000 / HumanEval 164 / MBPP 500 / ARC 3,548 / HellaSwag 10,042 → 총 69,645 항목.

**실측과 함정.** 첫 실행에서 수학 1.3% 가 제거됐는데, 상위 사례가 OEIS 정수열 페이지였다. `1 0 0 0 1 0 0 ...` 같은 13-gram 이 MATH 정답의 숫자 나열과 일치한 **거짓 양성**이다. 그래서 13-gram 안에 **서로 다른 단어가 8 개 미만이면 무시**하는 규칙을 인덱스와 문서 양쪽에 넣었다. 그 뒤 0.99% 가 제거됐고, 남은 사례는 AoPS 위키의 AMC/AIME 문제(MATH 벤치마크의 출처), Project Euler, 소수 판별 코드(MBPP/HumanEval), GSM8K 류 퍼즐로 **실제 오염**이다. 수학 도메인은 벤치마크와 출처가 겹치므로 1% 수준이 정상이고, 영어 일반 웹은 0.1% 미만을 기대한다.

**한계.** 바꿔 쓴(paraphrased) 문항은 못 잡는다. 번역된 문항(영어 MMLU 의 한국어 번역)도 못 잡는다. 제거율은 보통 0.01~0.1% 수준이며, `contaminated_samples.txt` 로 실제 무엇이 걸렸는지 확인한다. 평가 전용 데이터(향후 추가하는 벤치마크)는 학습 전 반드시 인덱스에 추가해야 한다.

---

## 9. Step 8-9 — Tokenizer v1

정제·dedup·decontam 이 끝난 `cleaned/<domain>_final` 에서 코퍼스를 만든다. 비율은 학습 mixture 와 같은 50/25/15/10. `--keep-prob` 로 문서를 띄엄띄엄 뽑아 특정 파일에 편중되지 않게 한다. 학습 후 `eval_tokenizer.py` 로 bytes/token 을 v0 과 비교한다. 기대: 한국어 bytes/token 이 4.56 보다 개선(코퍼스 비율 25% 가 제대로 반영되므로).

**이후 동결.** 토크나이저를 바꾸면 모든 샤드와 체크포인트가 무효가 된다. v1 이 확정되면 `tokenizer/researchlm_32k.json` 에 코퍼스 출처가 기록되고, 그 뒤로는 바꾸지 않는다.

---

## 10. Step 10 — 토크나이즈와 디스크

20B 토큰 × 2 B (uint16) = **40 GB** 의 `.bin`. `/storage` 에 충분하다. `tokenize_data.py` 는 문서를 스트리밍하므로 RAM 을 쓰지 않고, 16~32 프로세스로 약 100k 문서/분 이상 처리한다 (2,500 만 문서 → 수십 분).

---

## 11. 스케일링: 50M → 150M → 300M → 1B

### 11.1 설정 표

| | 50M | 150M | 300M | 1B |
|---|---:|---:|---:|---:|
| 파라미터 (실측) | 50.6M | 151.0M | 304.1M | 997.5M |
| Layers / d_model | 12 / 512 | 20 / 768 | 24 / 1024 | 21 / 2048 |
| Q / KV heads | 8 / 2 | 12 / 4 | 16 / 4 | 32 / 8 |
| FFN | 1,408 | 2,048 | 2,816 | 5,504 |
| Context | 2,048 | 2,048 | 4,096 | 4,096 |
| 학습 토큰 | 1B | 3B | 6B | 20B (→40B/80B) |
| tokens / step | 0.52M | 0.52M | 1.05M | 2.1M |
| micro-batch × accum | 16 × 8 | 16 × 8 | 8 × 16 | 4 × 64 |
| LR / 스케줄 | 1.5e-3 cosine | 1.0e-3 cosine | 6e-4 cosine | 4e-4 **WSD** |
| grad checkpointing | 아니오 | 아니오 | 예 | 예 |
| GPU 메모리 (실측) | 14.5 GB | 27.4 GB | 14.9 GB | 24.5 GB (mb 4, compile) / 29.6 GB (미컴파일) |
| 처리량 (2× RTX 8000) | 148k tok/s (pilot 실측) | 72k tok/s | 22k tok/s | 9~10k tok/s (compile) / 8.1k (미컴파일) |
| 예상 시간 | 약 2 h | 약 12 h | 약 3.3 일 | **24~27 일** |

### 11.2 왜 이렇게 정했나

- **배치 크기.** 모델이 클수록 gradient 노이즈를 견디는 "critical batch size"가 커진다 (McCandlish et al. 2018). 0.5M → 1M → 2M 토큰은 GPT-3 (0.5M~3.2M), Llama (4M) 의 범위 안이다. 너무 크면 같은 토큰으로 step 수가 줄어 토큰 효율이 떨어진다.
- **LR.** 폭(d_model)이 커질수록 최적 LR 이 내려간다 (μP: ∝ 1/width). 50M 에서 sweep 으로 최적을 찾고, 150M 은 ×(512/768) ≈ 0.67, 1B 는 ×0.25 수준으로 낮춘다. 표의 값은 그 비율을 따른 **초기 추정**이며 50M 결과로 교체한다.
- **WSD for 1B.** goal.md §2 의 "20B 후 loss 를 보고 40B·80B 로 연장" 을 하려면, cosine 은 총 step 을 처음부터 알아야 하지만 WSD 는 "stable" 구간의 체크포인트에서 decay 만 다시 돌리면 된다. 20B 시점의 decay 체크포인트로 평가하고, 더 가기로 하면 stable 체크포인트에서 재개한다.
- **Gradient checkpointing.** forward 에서 블록 활성값을 저장하지 않고 backward 때 다시 계산한다. 메모리 ≈ 활성값의 1/L 로 줄고 연산 ≈ 33% 증가. 1B × seq 4096 은 이것 없이 micro-batch 1 도 48 GB 에 빠듯하다. `ResearchLM.gradient_checkpointing = True` 가 `torch.utils.checkpoint` 로 블록을 감싼다 (KV cache 생성 시에는 비활성).
- **FP16 에서 1B.** 모델이 클수록 활성값·logit 폭주 위험이 커진다. QK-Norm, fp32 norm/RoPE/loss, GradScaler 가 방어선이고, 1B 에서는 **`z-loss`** (출력 logit 의 log-partition 을 0 근처로 묶는 항, PaLM) 추가가 유효한 보강책이다. 300M 에서 fp16 불안정이 관찰되면 넣는다.
- **150M/300M 의 역할.** 50M 의 결론(데이터 비율, QK-Norm, GQA 비율)이 크기를 올려도 유지되는지 확인하는 중간 점검이다. 150M 은 반나절, 300M 은 3 일 남짓이므로 1B 에 3~4 주를 쓰기 전의 보험이다.

### 11.3 1B 가 2× RTX 8000 에서 3~4 주인 이유와 선택지

6 × 1B × 20B = 1.2 × 10²⁰ FLOPs. RTX 8000 FP16 peak 130 TFLOPs × 2 × MFU 30% ≈ 78 TFLOPs → 18 일이 이론 하한이고, 실측 10k tok/s(compile, grad ckpt 의 33% 재계산 포함; MFU ≈ 23%)는 **약 24 일**이다. micro-batch 를 4 에서 더 키울 메모리 여유(24.5/48 GB)가 있지만 2M tokens/step 를 유지하려면 accum 과 함께 바꿔야 한다. 선택지:

1. 그대로 진행 (학교 서버가 그 기간 점유됨). 체크포인트가 500 step 마다 저장되므로 중단·재개는 자유롭다.
2. Server 1 (A10G 1 장)은 1B 를 돌리기에 더 느리므로 1B 는 Server 2 전용.
3. 클라우드 A100/H100 8 장 임시 사용: 같은 run 이 1~2 일. 코드는 DDP 그대로 동작(`torchrun --nproc_per_node=8`), bf16 자동 선택.

---

## 12. 실행 순서 (명령어)

모두 `conda activate researchlm`, 저장소 루트. 긴 작업은 `nohup ... > logs/xxx.log 2>&1 &` 또는 `tmux` 로 돌린다.

```bash
# ---------- Phase 1: 다운로드 (수 시간, 디스크 ~75 GiB) ----------
python scripts/download_data.py --config configs/data/download_v1.yaml --domain english
python scripts/download_data.py --config configs/data/download_v1.yaml --domain korean
python scripts/download_data.py --config configs/data/download_v1.yaml --domain math
python scripts/download_data.py --config configs/data/download_v1.yaml --domain code_meta
python scripts/fetch_stack_edu.py --config configs/data/download_v1.yaml --language all      # ~3 h

# ---------- Phase 1: 구조 분석 + 수동 검토 ----------
for d in english korean math; do
  python scripts/data_structure_analyzer.py --domain $d --input /storage/researchlm/raw/$d --sample 20000
done
python scripts/data_structure_analyzer.py --domain code --input /storage/researchlm/raw/code --sample 20000
#   -> /storage/researchlm/metadata/analysis/*.png, *_summary.json, *_examples.jsonl 검토

# ---------- Phase 2: 필터 ----------
for d in english korean math; do
  python scripts/filter_data.py --domain $d --input /storage/researchlm/raw/$d --out /storage/researchlm/cleaned/$d --workers 48
done
python scripts/filter_data.py --domain code --input /storage/researchlm/raw/code --out /storage/researchlm/cleaned/code --workers 48
#   -> cleaned/<d>/filter_stats.json, rejected_samples.txt 읽고 filters_v1.yaml 조정 후 필요하면 재실행

# ---------- Phase 2: dedup + decontamination ----------
python scripts/download_benchmarks.py
python scripts/decontaminate_data.py --build-index --index /storage/researchlm/metadata/benchmarks/ngram13.npy
for d in english korean math code; do
  python scripts/dedup_data.py --domain $d --input /storage/researchlm/cleaned/$d --out /storage/researchlm/cleaned/${d}_dedup --workers 48
  python scripts/decontaminate_data.py --domain $d --index /storage/researchlm/metadata/benchmarks/ngram13.npy \
      --input /storage/researchlm/cleaned/${d}_dedup --out /storage/researchlm/cleaned/${d}_final --workers 48
done

# ---------- Phase 3: tokenizer v1 (정제 코퍼스, 동결) ----------
python scripts/train_tokenizer.py \
  --input english=/storage/researchlm/cleaned/english_final --input korean=/storage/researchlm/cleaned/korean_final \
  --input code=/storage/researchlm/cleaned/code_final --input math=/storage/researchlm/cleaned/math_final \
  --weights english=0.5,korean=0.25,code=0.15,math=0.10 \
  --keep-prob english=0.05,korean=0.1,code=0.1,math=0.3 --target-chars 2000000000 \
  --out-prefix /storage/researchlm/tokenizer/researchlm_32k --threads 64
python scripts/eval_tokenizer.py --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
  --input english=/storage/researchlm/cleaned/english_final --input korean=/storage/researchlm/cleaned/korean_final \
  --input code=/storage/researchlm/cleaned/code_final --input math=/storage/researchlm/cleaned/math_final --limit 3000

# ---------- Phase 3: 토크나이즈 ----------
python scripts/tokenize_data.py --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
  --input english=/storage/researchlm/cleaned/english_final --input korean=/storage/researchlm/cleaned/korean_final \
  --input code=/storage/researchlm/cleaned/code_final --input math=/storage/researchlm/cleaned/math_final \
  --out-dir /storage/researchlm/tokenized/v1 --val-fraction 0.005 --workers 48

# ---------- Phase 5: 50M 정식 + ablation ----------
torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_50m.yaml --train configs/train/50m_server2.yaml
for lr in 6e-4 1e-3 3e-3; do
  torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_50m.yaml \
    --train configs/train/50m_server2.yaml --set run_name=50m-lr$lr --set lr=$lr --set max_steps=1000
done
sed 's/qk_norm: true/qk_norm: false/' configs/model/researchlm_50m.yaml > configs/model/researchlm_50m_noqk.yaml
torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_50m_noqk.yaml \
  --train configs/train/50m_server2.yaml --set run_name=50m-noqknorm --set max_steps=1000
torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_50m.yaml \
  --train configs/train/50m_server2.yaml --set run_name=50m-koheavy --set "mixture={english: 0.30, korean: 0.45, code: 0.15, math: 0.10}" --set max_steps=1000

# ---------- Phase 6-8 ----------
torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_150m.yaml --train configs/train/150m_server2.yaml
torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_300m.yaml --train configs/train/300m_server2.yaml
torchrun --standalone --nproc_per_node=2 scripts/train.py --model configs/model/researchlm_1b.yaml   --train configs/train/1b_server2.yaml
# 중단 후 재개:  ... --resume latest
# 1B 를 40B 로 연장: stable 구간 체크포인트에서  --resume <step_dir> --set max_steps=20000
```

---

## 13. 데이터 파이프라인 실패 진단

| 증상 | 원인 | 조치 |
|---|---|---|
| 다운로드가 느림 / 끊김 | HF 익명 rate limit | `export HF_TOKEN=...` 설정, `--workers` 줄이기. 재실행하면 이어받음 |
| `fetch_stack_edu` 에 `missing` 다수 | SWH 에 없는 blob (드묾) | 1% 미만이면 무시. 많으면 네트워크/S3 리전 확인 |
| 필터 유지율이 50% 미만 | 임계값이 데이터 분포와 안 맞음 | `filter_stats.json` 상위 사유 → `rejected_samples.txt` 읽기 → 임계값 완화 |
| 한 규칙이 20% 이상 거부 | 특징 구현 오류 가능 | 예문으로 특징값 직접 계산해 확인 (§6 교훈) |
| dedup 이 메모리 초과 | 문서 수 > 1 억 | 도메인을 파일 묶음으로 나눠 실행, 또는 `--num-perm 64` |
| near-dup 비율이 0% | 샤드가 너무 작거나 밴드 설정 | 100 문서 이상인지, `--bands 16` 인지 확인 |
| decontam 제거율 > 1% | 벤치마크 텍스트에 흔한 문구 포함 | `contaminated_samples.txt` 확인, `--min-hits 2` |
| 토크나이즈 샤드 토큰 수 ≠ 예상 | 예산 추정(chars/token) 오차 | `tokenized/v1/manifest.json` 로 실제 토큰 확인 후 mixture 재계산 |
| 학습 중 특정 도메인 epoch > 4 | 그 도메인 토큰 부족 | `describe()` 출력의 "epochs over the run" 확인, 추가 다운로드 |

---

## 14. 추가된 파일

```text
researchlm/data/
  io.py            parquet / jsonl / jsonl.zst 읽기·쓰기, row-group 청크 분할
  quality.py       텍스트·코드 품질 특징 (Gopher / C4 / StarCoder 휴리스틱)
  filters.py       YAML 규칙 기반 도메인 필터
  dedup.py         sha1 exact, MinHash + 정렬 기반 LSH, union-find
  decontaminate.py 13-gram 인덱스
scripts/
  download_data.py           HF parquet 예산 다운로드
  fetch_stack_edu.py         Software Heritage S3 코드 수집 (resume)
  data_structure_analyzer.py 특징 + 메타데이터 분석, 그래프
  filter_data.py             병렬 필터, 거부 사유 통계·샘플
  dedup_data.py              exact + near dedup
  download_benchmarks.py     8 개 벤치마크 수집
  decontaminate_data.py      인덱스 생성 + 제거
configs/
  data/download_v1.yaml      다운로드 예산, Stack-Edu 언어 비율
  data/filters_v1.yaml       도메인별 규칙 (history 기록)
  model/researchlm_{150m,300m,1b}.yaml
  train/{150m,300m,1b}_server2.yaml
tests/test_data_pipeline.py  9 개 테스트
```
