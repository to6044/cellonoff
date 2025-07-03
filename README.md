# EXP3 Cell On/Off 시나리오 구현 가이드

## 개요
이 문서는 네트워크 시뮬레이터에 EXP3 (Exponential-weight algorithm for Exploration and Exploitation) 알고리즘 기반의 cell on/off 시나리오를 추가하는 방법을 설명합니다.

## 주요 기능

### 1. EXP3 알고리즘 구현
- **Multi-armed bandit 문제**: k개의 기지국 중 n개를 끄는 최적의 조합 찾기
- **보상 함수**: 네트워크 효율성 (cell_throughput / cell_power)
- **온라인 학습**: 이벤트 기반으로 실시간 학습 및 적응

### 2. 초기 탐색 (Warm-up)
- ε-greedy 방식으로 초기 100 에피소드 동안 무작위 탐색
- `warm_up` 파라미터로 활성화/비활성화 가능
- 충분한 baseline 데이터 확보 후 EXP3 학습 시작

### 3. 학습 결과 시각화
- **가중치 변화**: 각 arm의 선택 확률 변화 추적
- **네트워크 효율성**: 시간에 따른 효율성 개선 그래프
- **최적 구성**: 가장 높은 확률을 가진 cell on/off 조합
- **상세 분석**: 수렴 분석, 보상 분포, 성능 개선 지표

## 설치 방법

### 1. kiss.py 파일 수정

```python
# 1) Import 추가 (파일 상단)
from itertools import combinations

# 2) EXP3CellOnOff 클래스 추가 (전체 클래스 코드 복사)
class EXP3CellOnOff(Scenario):
    # ... (제공된 전체 클래스 코드)

# 3) main() 함수에서 시나리오 정의 추가
exp3_cell_onoff = EXP3CellOnOff(
    sim,
    k_cells=config_dict.get("exp3_k_cells", None),
    n_cells_off=config_dict.get("exp3_n_cells_off", 5),
    interval=base_interval,
    eta=config_dict.get("exp3_eta", 0.1),
    warm_up=config_dict.get("exp3_warm_up", True),
    warm_up_episodes=config_dict.get("exp3_warm_up_episodes", 100),
    output_dir=data_output_logfile_path,
    delay=scenario_delay
)

# 4) 시나리오 활성화 조건 추가
elif scenario_profile == "exp3_cell_onoff":
    exp3_cell_onoff.setup_energy_models(cell_energy_models_dict)
    sim.add_scenario(scenario=exp3_cell_onoff)
```

### 2. JSON 설정 파일 생성

`data/input/exp3_cell_onoff/exp3_config.json` 파일 생성:

```json
{
    "experiment_description": "exp3_cell_onoff",
    "scenario_profile": "exp3_cell_onoff",
    "until": 200,
    "nues": 400,
    
    "exp3_k_cells": null,
    "exp3_n_cells_off": 5,
    "exp3_eta": 0.1,
    "exp3_warm_up": true,
    "exp3_warm_up_episodes": 100,
    
    // ... 기타 필요한 설정들
}
```

### 3. 분석 스크립트 설치

`analyze_exp3_results.py` 파일을 프로젝트 디렉토리에 저장

## 사용 방법

### 1. 시뮬레이션 실행

```bash
cd /path/to/KISS
python run_kiss.py -c data/input/exp3_cell_onoff/exp3_config.json
```

### 2. 결과 확인

시뮬레이션 완료 후 `data/output/exp3_cell_onoff_YYYYMMDD_HHMMSS/` 폴더에서:
- `exp3_weights_history.json`: 학습 데이터
- `exp3_learning_results.png`: 학습 과정 시각화
- `exp3_best_configuration.png`: 최적 구성
- `*.tsv`: 시뮬레이션 로그

### 3. 상세 분석

```bash
python analyze_exp3_results.py --results_dir data/output/exp3_cell_onoff_YYYYMMDD_HHMMSS
```

## 파라미터 설명

### 필수 파라미터
- **k_cells**: 고려할 cell 인덱스 리스트 (null = 모든 cell)
- **n_cells_off**: 끌 cell 개수
- **eta**: EXP3 학습률 (0.01 ~ 0.5, 기본값: 0.1)

### 선택 파라미터
- **warm_up**: warm-up 단계 사용 여부 (기본값: true)
- **warm_up_episodes**: warm-up 에피소드 수 (기본값: 100)
- **interval**: 결정 주기 (기본값: 1.0)
- **delay**: 시작 지연 시간 (기본값: 0.0)

## 알고리즘 동작 원리

### 1. 초기화
- 가능한 모든 cell 조합 (arms) 생성: C(k, n)
- 각 arm에 동일한 초기 가중치 부여

### 2. Warm-up 단계 (선택적)
- 무작위로 arm 선택하여 baseline 성능 측정
- 각 arm의 기본 보상 정보 수집

### 3. EXP3 학습 단계
- 확률 분포에 따라 arm 선택
- 선택된 cell들을 on/off
- 네트워크 효율성 측정 (보상 계산)
- 가중치 업데이트: w_i = w_i * exp(η * r_i / p_i)
- 확률 분포 재계산

### 4. 수렴
- 최적 arm의 선택 확률이 증가
- 네트워크 효율성 최대화

## 결과 해석

### 1. 학습 곡선
- **Efficiency over time**: 상승 곡선은 성공적인 학습을 의미
- **Entropy**: 감소하는 entropy는 특정 arm으로의 수렴을 의미

### 2. 최적 구성
- **Best configuration**: 가장 높은 확률을 가진 arm
- **Cell OFF frequency**: 각 cell이 꺼진 빈도

### 3. 성능 지표
- **Initial vs Final efficiency**: 개선율 확인
- **Throughput vs Power trade-off**: 효율성 최적화 확인

## 문제 해결

### 1. 수렴하지 않는 경우
- `eta` 값 조정 (너무 크면 불안정, 너무 작으면 느린 학습)
- `warm_up_episodes` 증가
- 시뮬레이션 시간 (`until`) 증가

### 2. 성능이 개선되지 않는 경우
- `n_cells_off` 값 조정
- 네트워크 부하 확인 (UE 수)
- cell 간 간섭 확인

### 3. 메모리 문제
- arm 수가 너무 많은 경우 `k_cells` 제한
- 히스토리 저장 주기 조정

## 확장 가능성

### 1. 보상 함수 커스터마이징
```python
# calculate_network_efficiency() 메서드 수정
def calculate_network_efficiency(self):
    # 커스텀 메트릭 추가
    qos_metric = self.calculate_qos()
    fairness_metric = self.calculate_fairness()
    
    # 복합 보상 함수
    reward = alpha * efficiency + beta * qos_metric + gamma * fairness_metric
    return reward
```

### 2. 동적 arm 생성
```python
# 시간대별로 다른 cell 조합 고려
def generate_time_aware_arms(self, current_time):
    if is_peak_hour(current_time):
        return self.peak_hour_arms
    else:
        return self.off_peak_arms
```

### 3. 다중 목표 최적화
```python
# 파레토 최적 솔루션 찾기
def multi_objective_exp3(self):
    # throughput, power, fairness 등 다중 목표 고려
    pass
```

## 참고 문헌
1. Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). The nonstochastic multiarmed bandit problem.
2. 5G Network Energy Efficiency Optimization papers
3. KISS/AIMM Simulator Documentation
