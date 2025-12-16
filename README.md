
# Chatterbox TTS Fine-tuning (Korean)

본 저장소는 **Chatterbox 공식 TTS 모델**을 기반으로 한국어 음성 생성을 위해 파인튜닝을 수행한 코드와 실험 파이프라인을 포함한다.  
Emilia Dataset의 한국어 semantic token을 활용하여 semantic-level 파인튜닝을 수행하고, 파인튜닝된 모델(`final.pt`)을 이용해 한국어 음성을 생성한다.

---

## Overall Pipeline

본 레포에서의 전체 파이프라인은 다음과 같다.

1. `dataset_loader_emilia.py`  
   - Emilia Dataset에서 한국어(KR) semantic token 스트리밍 로드

2. `finetune_t3_ko.py`  
   - 한국어 semantic token을 사용한 T3 모델 파인튜닝
   - 학습 완료 후 `final.pt` 생성

3. `inference_semantic.py`  
   - 파인튜닝된 `final.pt`를 로드하여 한국어 음성 생성
   - 필요 시 speaker reference(`voice_sample.wav`)를 활용한 보이스 클로닝 수행

---

## What is included

- Emilia Dataset 기반 한국어 semantic loader
- T3 semantic 모델 파인튜닝 코드
- 파인튜닝된 checkpoint (`final.pt`) 생성
- Chatterbox 공식 inference 코드와 결합한 음성 생성
- Optional speaker reference 기반 voice cloning

---

## My Environment

- Kubernetes Pod 기반 실행 환경
- NVIDIA CUDA 12.x + Ubuntu 22.04
- 단일 GPU 환경 (A100 40GB)
- Persistent Volume 기반 작업 디렉토리
- PyTorch 기반 학습 및 추론

---

## Environment Setup

### 1. Python 가상환경 생성 및 활성화

```
bash
python -m venv .env
source .env/bin/activate
```

### 2. 시스템 의존성 설치

```
sudo apt update
sudo apt install ffmpeg
```

### 3. Python 패키지 설치

```
pip install -r requirements.txt
```

---

## Fine-tuning  

Dataset  
- 사용 데이터셋: Emilia Dataset
- 접근 방식: HuggingFace streaming=True
- 사용 조건:
  - language == "KR"
  - semantic token이 존재하는 샘플만 사용
  
데이터 로딩은 dataset_loader_emilia.py에서 수행된다.

---

## Training Execution

파인튜닝은 아래 스크립트를 통해 수행된다.

```
python finetune_t3_ko.py
```

## Output Checkpoint

학습 완료 후 다음 경로에 모델이 저장된다.

```
ckpt/final.pt
```

final.pt는 항상 finetune_t3_ko.py가 실행된 디렉토리 기준의 ckpt/ 하위에 저장되도록 고정되어 있다.  
(경로를 인자로 변경하는 구조는 아니다.)

---

## Inference (Speech Generation)

파인튜닝된 모델을 이용한 음성 생성은 inference_semantic.py를 통해 수행된다.

```
python inference_semantic.py
```

- 기본 텍스트는 스크립트 내부에 정의되어 있으며,
- ckpt/final.pt가 존재할 경우 해당 모델을 로드한다.
- 로드에 실패할 경우 Chatterbox 공식 multilingual 모델로 fallback 된다.
  
생성된 음성은 다음 파일로 저장된다.
```
output_ko.wav
```

---

## Speaker Reference (Voice Cloning)

voice_sample.wav 파일은 optional speaker reference로 사용된다.  
- 코드 레벨에서는 필수 요소는 아니며,
- 제거하더라도 파인튜닝 및 inference는 정상 동작한다.
  
다만, 본 프로젝트에서는 voice cloning 실험을 포함하는 것을 요구사항으로 두었기 때문에
README 및 실험 구성에는 speaker reference 사용을 포함하여 설명한다.

