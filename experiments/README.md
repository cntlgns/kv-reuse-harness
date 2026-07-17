# Experiments

하네스 × KV-cache reuse co-design 연구의 실험 코드와 기록.

## 구성

```
experiments/
├── scripts/            # 실험 유틸리티 (버전 기록, 부하 생성, 집계 등)
├── vllm_plugins/       # vLLM entry-point 플러그인 (서버측 요청 원장)
├── replay/             # (Phase 1) 궤적 수집 + replay 도구
└── results/            # 실험 결과 — git 밖 (.gitignore), 요약/figure만 커밋
```

## Exp-1: 컨텍스트 관리 모드 × 벤치마크 (2026-07-14)

Qwen3-Coder-30B-A3B (vLLM, cobra:8123, ctx 262144) 로 SSA의 컨텍스트 관리 정책을 비교.
**756 런 = 4 arm × (SWE-Bench Verified 100 + Terminal Bench 2 전수 89).**

| Arm | 정책 | 오버라이드 |
|-----|------|-----------|
| A  | 풀 컨텍스트 (append-only) | (config 기본값: win_len=100000, budget=262144×0.9) |
| B1 | 토큰 예산 64k | `agent.context_window.max_model_len=65536` |
| B2 | 토큰 예산 32k | `agent.context_window.max_model_len=32768` |
| C  | 슬라이딩 윈도우 | `agent.conversation_manager.win_len=60` |

- 인스턴스: `sbv_sample_100.txt` (seed 42, sorted 500 중 100), `tb2_all_89.txt` (전수)
- config: `sbv_openai_qwen3_coder.yaml`, `tb2_openai_qwen3_coder.yaml` (샘플링 파라미터 전 arm 고정)
- 실행: `scripts/submit_all.sh` — 인스턴스 목록을 12샤드×2벤치로 나눠 david/tao/ruby에
  packed sbatch (hal-harness swebench_matrix 패턴). 샤드 내부는 인스턴스-major
  (인스턴스당 4 arm 순차 실행 후 docker 이미지 정리). 재실행 시 DONE 마커로 스킵.
- 진행 확인: `scripts/status.sh` / 집계: `scripts/aggregate.py` →
  `results/ssa/summary.csv` + arm별 `preds_sbv_<ARM>.jsonl` (SWE-bench 공식 채점용)
- tb2 채점은 하네스 내장 (`reward.txt`), sbv는 preds를 SWE-bench harness로 별도 채점

## Cost 측정 인프라 (2026-07-15)

서빙 비용(tasks/hour/GPU, GPU-sec per solved task)을 재기 위한 계측 4종.
측정 단위는 요청이 아니라 **태스크(인스턴스 완주)** — 컨텍스트를 줄여 턴을
더 쓰는 정책은 tasks/hour 하락과 turns_per_task 증가로 자동으로 드러난다.

| 층 | 구성요소 | 산출물 |
|----|---------|--------|
| 클라이언트 | `SROpenAIModel` per-call 텔레메트리 (`agent.invoker_params.request_log=true`로 opt-in; 기본 off) | 런 디렉토리마다 `requests.jsonl` — request_id, 송신/TTFT/종료 타임스탬프, usage |
| 서버 원장 | `vllm_plugins/ssa_request_ledger` (vLLM stat-logger 플러그인, `SSA_REQUEST_LEDGER_DIR`로 opt-in) | `results/serve/<jobid>/ledger-0.jsonl` — 요청별 queue/prefill/decode 시간, cached 토큰 |
| 엔진 시계열 | `scripts/scrape_metrics.py` + nvidia-smi (serve sbatch가 자동 기동) | `metrics_scrape.jsonl`, `gpu_samples.csv` |
| 부하 생성 | `scripts/load_driver.py` — closed-loop, 항상 M개 태스크 유지, 매니페스트 순환 | `results/load/<name>/` (`events.jsonl`, `tasks/NNNN_<id>/`) |

조인 키: 클라이언트가 매 호출 `request_id = {bench}.{arm}.{task}.{run_token}.c{seq}`를
extra_body로 보내고, 서버 원장에는 `chatcmpl-` 접두사가 붙어 기록된다.

측정 셀 하나(= bench × arm × M) 실행 → 분석:

```bash
# 서버는 serve_qwen3_coder_h100.sbatch 로 기동 (원장·스크랩 사이드카 자동)
python3 experiments/scripts/load_driver.py \
    --bench sbv --arm C -M 8 --duration 7200 \
    --manifest experiments/sbv_sample_100.txt \
    --out experiments/results/load/sbv-C-M8-r0 --prepull

python3 experiments/scripts/analyze_load.py \
    --run experiments/results/load/sbv-C-M8-r0 \
    --ledger experiments/results/serve/<jobid>/ledger-0.jsonl \
    --scrape experiments/results/serve/<jobid>/metrics_scrape.jsonl \
    [--solved-map solved.json]   # sbv: 오프라인 채점 결과 주입; tb2는 reward.txt 자동
```

`analyze_load.py`는 warmup(기본 600s) 이후 ~ launch_stopped 사이의 steady-state
창만 집계한다: tasks/hour/GPU, **GPU-sec/solved-task**(품질 보정 비용),
turns/토큰 일량 per task (`tasks.csv`), TTFT/TPOT 백분위, 시간평균 배치 크기,
KV 사용률, preemption. 조건 간 비교 시 서버 재시작으로 캐시 상태를 초기화할 것.

주의: 클라이언트 usage의 `cached_tokens`는 서버가 `--enable-prompt-tokens-details`로
떠 있어야 채워진다 (2026-07-15 이전에 뜬 서버는 플래그 없음 — metrics.json의
cache_hit_rate=0.0은 이 때문이며, prefix cache 자체는 정상 동작했음).

## Exp-2: 서빙 cost 측정 — arm A vs C × M 스윕 (2026-07-16)

sbv 50-인스턴스(`sbv_load_50.txt`), closed-loop M ∈ {8,16,32}, 셀당 60분 창(워밍업
10분 제외), kill-stop. 서버: cobra H100×4 (A군 잡 45608, C군 잡 45617 — arm 전환 시
재시작으로 캐시 초기화). 클라이언트: david 전용. 결과: `results/load/sbv-{A,C}-M*-r0/`.

핵심 결과 (tasks/hour/GPU): A = 24.0/33.3/43.2 vs C = 17.1/22.5/27.3 —
**슬라이딩 윈도우(C)가 전 구간 29–37% 더 비쌈**. 원인: prefix cache 히트
98%→53%로 붕괴, 태스크당 실제 prefill 계산 토큰 15배(27k→400k). 컨텍스트는
30% 짧아졌지만(26k→18k) decode 절감은 4%에 불과. C의 KV 점유는 낮음(max
42%→29%). prefill 손해는 자기 스팬(LLM 시간의 4–7%)이 아니라 **배치 전체의
decode 감속**(chunked-prefill 간섭, TPOT 22→29ms@M16)으로 전이됨.

고부하 연장 (M∈{64,96}, 30분 셀, 100-인스턴스, A=잡 45651 / C=잡 45740):
A = 51.0 (M64) → **16.5 (M96: KV 100% 도달 + preemption → 붕괴)**;
C = 25.5 (M64) → 25.5 (M96: KV max 88%, preemption 0, 유지) → **M96에서 교차
(C 25.5 > A 16.5)**. 해석: A의 벽은 KV 메모리(M≈90), C의 벽은 캐시 파괴발
prefill 컴퓨트(TPOT 52→132ms, ~25에서 조기 plateau). 비프리픽스 재사용 =
C의 컴퓨트 벽 제거 + 낮은 KV 점유 유지 → 전 구간 우위 가설. 주의: M96
셀들은 30분 창이라 램프 영향 큼(완료 22/34개) — 인용 전 창 연장 반복 필요.
SLO 관점에서는 M≥64는 양쪽 다 TTFT p99 20s+ 위반 영역.

## Exp-2b: M64/M96 재측정 — 정상성 확보 창 (2026-07-16 오후)

아침 프로브가 `check_stationarity.py` 진단에서 비정상 판정(Little 비 0.22–0.49,
첫 코호트 3–5%만 워밍업 내 완주)이라 재측정. M64=2h 셀(분석 워밍업 40분),
M96=4h 셀(워밍업 60분), **50-인스턴스 매니페스트로 통일**(M8–32 커브와 동일
워크로드). 병렬 2-lane: A=신규 서버 45748(cobra:8124)+david, C=45740(cobra:8123,
아침 C군 서버 유지 — 같은 arm이라 재시작 불요)+tao. 결과: `sbv-{A,C}-M{64,96}-r1/`.

| tph/gpu | M8 | M16 | M32 | M64 | M96 |
|---------|-----|-----|-----|-----|-----|
| A | 24.0 | 33.3 | 43.2 | **43.9** | **20.6** |
| C | 17.1 | 22.5 | 27.3 | 27.0 | **31.5** |

전 셀 정상성 통과(Little 비 0.90–0.99, r1 기준; A-M96은 워밍업 60→90분 민감도
+3%뿐). **M96 교차 확정: C 31.5 > A 20.6** (GPU-s/task 114 vs 175).

A-M96 붕괴 메커니즘 정정: preemption 폭풍(16회뿐)이 아니라 **캐시 축출 thrash +
admission 병목**. running~81개(×26k)가 풀 2.34M를 pin → 턴 사이 재히트를 기다리던
refcount-0 블록이 LRU 축출 → 히트 98%→**17%**, prefill 재계산 31.7k→**1.18M/task**
(37×), queued p50 11.7s, TPOT 63→266ms, TTFT p50 15.9s. 즉 M96의 A는 "풀 길이
컨텍스트를 가진 C"로 퇴화. C는 M96에서도 스케일 지속(KV max 0.79, preempt 0,
queued p50 0.2s).

단 **전역 최적은 여전히 A@M64**(82 GPU-s/task; C 최적 M96 114보다 28% 쌈) —
A의 커브가 M64 직후 절벽(메모리 벽 M≈90 = 2.34M/26k 산수와 일치)인 반면 C는
강건. 비프리픽스 재사용의 목표 재정식화: **C의 강건성 + A의 캐시 경제성**
(C의 재-prefill 613k/task를 재사용으로 제거하면 M96+에서 양쪽 벽 모두 회피).
채점 완료(Exp-2c에서): r1 셀 preds 회차별 채점 — A 33.5%(4라운드; Exp-1 동일
서브셋 34%), C 40.5%(4라운드; 기준 45%, 라운드 분산 ±7%p) → **M96 thrash에서도
품질 저하 없음** = 부하 셀에서 품질 수확 방법론 유효.
창 내 rc≠0: A 5/247, C 3/378 (~1–2%, tasks_rc0로 분리 집계됨).

## Exp-2c: 윈도우 스윕(win60/30/10) + tb2 비용 셀 + 품질 동시 수확 (2026-07-17)

캠페인 `scripts/campaign_20260717.sh` (+ `_chainC_recovery.sh`): 2-lane
(david↔8124, tao↔8123), arm 그룹 전환 시에만 서버 재시작, 셀 dur 7200s(분석 창
~110분). 신규 arm `win30`/`win10` = `agent.conversation_manager.win_len=30/10`
(load_driver.py ARM_OVERRIDES). 품질은 셀 완주 태스크에서 수확: sbv는
`build_load_preds.py`로 인스턴스별 k번째 완주를 round-k preds로 분리 →
`score_load_preds.sbatch`(공식 harness) 채점, tb2는 reward.txt 자동.
결과: `results/load/{sbv,tb2}-{A,C,win30,win10}-M*`, 채점 `results/load/quality/`.
사고 기록: C-lane 서버 교체 순간 타 사용자 잡이 GPU 선점 → 복구 스크립트로
체인A 종료 후 재개(win10 셀들이 늦게 실행된 이유).

### 비용 — sbv 윈도우 스윕 (tph/gpu, 장창)
| arm | M64 | M96 | cached% | prefill/task | 특이사항 |
|-----|-----|-----|---------|--------------|----------|
| A | **43.9** | 20.6 | 98→17% | 27k→1.18M | M96 KV thrash 붕괴 |
| C(win60) | 27.0 | **31.5** | ~50% | ~610k | M96 교차, KV 0.79 |
| win30 | 14.7 | 16.8 | 37–39% | 570–680k | |
| win10 | 3.7 | 4.1 | 43–47% | 1.8M+ | **21–23% 비종결(500콜 상한)**, 턴 423–429 |

### 비용 — tb2 (M32, 89전수, 110분 창)
A **21.7**(KV max 0.99 — tb2는 A의 메모리 벽이 M32 직후) > C 15.3(KV 0.29) >
win30 10.2 > win10 5.7(실패 46%, 턴 442). 윈도우 축소 = 비용 단조 악화가
양 벤치에서 재현. win10의 벽은 캐시도 KV도 아닌 **비종결(턴 폭발)**.

### 품질 (부하 셀 수확, sbv=공식 채점 / tb2=reward.txt macro pass@1)
| | A | C(win60) | win30 | win10 |
|---|---|---|---|---|
| sbv-100 | 41%* | **49%*** | 38% (39/37, 2라운드) | 24% (25/22) |
| tb2-89 | 15.0% | 15.9% | 15.2% | **11.0%** (73/89만 완주) |

*Exp-1 값. sbv에서 win60이 품질 정점, 그 아래는 품질도 하락. tb2는 A/C/win30
품질 둔감(15–16%) — 비용만 갈림.

### 사용자 과금 비용 (load_50 1회 완주, M64 셀, P_in=$1/M·P_out=$5/M, δ=P_cache/P_in)
| arm | input | cached% | δ=1 | δ=0.25 | $/solved(δ=.25) |
|-----|-------|---------|-----|--------|-----------------|
| A | 74.8M | 97.9% | $78.6 | **$23.7** | **$1.41** |
| C(win60) | 64.6M | 51.8% | $68.9 | $43.8 | $2.25 |
| win30 | 61.0M | 37.6% | $66.8 | $49.6 | $2.75 |
| win10 | 166.8M | 44.3% | $181.8 | $126.4 | $10.54 |

캐시 무할인(δ=1)에서만 win30 최저; 현실적 할인(δ≤0.25)에선 A 압승. win10은
프롬프트가 가장 짧은데도 턴 폭발로 총 input이 A의 2.2배.

### 종합
품질 축 최적=win60(sbv +8%p), 비용 축 최적=A(정상 부하) — 단 A는 KV 벽
(sbv M≈90, tb2 M≈32)에서 붕괴하고 win60은 캐시 파괴로 항상 29–45% 비쌈.
윈도우를 60 미만으로 줄이면 품질·비용·안정성 모두 단조 악화(win10은 파탄).
→ 비프리픽스 KV 재사용이 "win60의 품질+강건성"과 "A의 캐시 경제성"을 동시에
잡는 유일한 경로라는 동기가 4-arm × 2-bench로 완결.

## 관련 레포

- 하네스 (이 레포): `cntlgns/kv-reuse-harness` — 편집 정책, 실험 실행
- vLLM 포크: `cntlgns/vllm`, `research/base` 브랜치 (v0.24.0 기반) — 서빙 엔진 수정
  - 로컬: `../vllm` (기본값, `VLLM_ROOT`로 재지정 가능)
  - LMCache 0.5.1과 짝 (non-prefix reuse/blending 평가용)

두 레포 모두 **동결** 상태로 유지: upstream을 정기적으로 당겨오지 않고,
필요한 버그픽스만 cherry-pick 후 베이스라인 재실행.

## 재현성 규칙

모든 실험 결과 디렉토리에 `versions.json`이 있어야 한다:

```bash
./experiments/scripts/record_versions.sh <output_dir>
```

harness/vllm 두 레포의 commit SHA와 dirty 여부를 기록한다.
dirty 상태로 돌린 실험은 재현 불가능하므로, 본 실험 전에는 반드시 커밋할 것.

마일스톤마다 두 레포에 같은 이름의 태그를 찍는다: `exp/YYYYMMDD-<name>`
