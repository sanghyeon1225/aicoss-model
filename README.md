# 알츠하이머병 분류를 위한 멀티모달 어텐션

논문 Multimodal Attention-based Deep Learning for Alzheimer's Disease Diagnosis
 의 코드입니다.

# 데이터셋

Alzheimer's Disease Neuroimaging Initiative (ADNI) 데이터셋
을 사용하여 결과를 제공합니다.
이 데이터는 본 저장소에 포함되어 있지 않으며, ADNI에서 직접 요청해야 합니다.

# 요구사항

Python 3.7.4 (이상 버전)
Tensorflow 2.6.0
이 저장소에서 사용된 모든 패키지의 자세한 목록은 general/requirements.txt에서 확인할 수 있습니다.

# 설명

본 연구에서는 ADNI로부터 유전학, 임상, 영상 데이터를 활용하여 알츠하이머병을 탐지하기 위한 다중 모달(multi-modal), 다중 클래스(multi-class), 어텐션 기반 딥러닝 프레임워크를 제시했습니다.

<img src="https://user-images.githubusercontent.com/35315239/187262625-0f980b94-7cce-49ec-9041-421e56b67ecd.png" width="600">

이 저장소에는 위 논문의 코드가 포함되어 있습니다.
위 모델 아키텍처는 training/train_all_modalities.py에 구현되어 있습니다.

# 전처리

환자 ID와 진단 리스트를 생성하려면 general/diagnosis_making.ipynb 노트북을 실행하세요.

임상 데이터를 전처리하려면 preprocess_clincal/create_clinical_dataset.ipynb 노트북을 실행하면 필요한 데이터가 포함된 CSV 파일이 생성됩니다.

위 스크립트에서 사용되는 CSV 파일은 ADNI에서 직접 확보해야 하므로 노트북에는 포함되어 있지 않습니다.

영상 데이터를 전처리하려면, 먼저 preprocess_images.py를 실행하고 이미지가 저장된 디렉토리를 인자로 전달하세요.
그 후, 해당 스크립트가 생성한 파일을 사용하여 preprocess_images/splitting_image_data.ipynb 노트북을 실행하면 데이터를 학습용과 테스트용으로 분할할 수 있습니다.

유전학 데이터(SNP)를 전처리하려면, 먼저 ADNI에서 VCF 파일을 확보해야 합니다.
그런 다음, vcftools 패키지를 사용하여 선택한 기준(하디-바인베르크 평형, 유전자형 품질, 소수 대립유전자 빈도 등)에 따라 파일을 필터링하세요.
AD 관련 유전자에 따라 VCF 파일을 추가로 필터링하려면 filter_vcfs.py 스크립트를 실행하세요.
그 후, 모든 유전학 파일을 하나로 합치려면 concat_vcfs.py를 실행하세요.
마지막으로, 특징(feature) 수를 추가로 줄이려면 create_genetic_dataset.ipynb 노트북을 실행하세요.
관련 스크립트는 모두 preprocess_genetic 폴더에 있습니다.

# 학습 및 평가

단일 모달리티 모델(uni-modal baseline)을 학습 및 평가하려면 train_clinical.py, train_genetic.py, 또는 train_imaging.py를 실행하세요.
멀티모달 아키텍처를 학습 및 평가하려면 train_all_modalities.py를 실행하세요.

# 크레딧

이 저장소의 일부 구조는 https://github.com/soujanyaporia/contextual-multimodal-fusion
 에서 가져왔습니다.

# 저자

Michal Golovanevsky