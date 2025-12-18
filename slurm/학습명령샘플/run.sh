#!/bin/bash
#SBATCH --job-name=gpu_train         # 작업 이름
#SBATCH --nodes=2                   # 2개의 노드 사용
#SBATCH --ntasks=2                  # 작업 2개 실행 (노드당 하나)
#SBATCH --ntasks-per-node=1         # 노드당 1작업
#SBATCH --gpus-per-task=1           # 작업당 GPU 1개 할당
#SBATCH --time=02:00:00             # 최대 실행 시간 2시간
#SBATCH --output=gpu_train_output.log # 표준 출력 로그 파일
#SBATCH --error=gpu_train_error.log  # 표준 오류 로그 파일

# Python 가상 환경 활성화 (필요시)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch  # 활성화된 conda 환경에 필요한 패키지가 있어야 함

# Python 스크립트 실행 (딥러닝 모델 학습)
srun --unbuffered python test3.py

