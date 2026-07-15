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
