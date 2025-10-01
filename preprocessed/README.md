## Clinical

문자열 형 = 없음
수치형 = 결측치를 포함한 원-핫 인코딩
숫자형 = 결측치를 포함하는 행 제거 + Min-Max 정규화

X의 feature 수: 121개

## Image

가공되지 않은 raw 이미지 => 72*72*3 로 슬라이싱

mri_meta.csv는 clinical 파일에 있으며
mri_meta.pkl은 images 파일에 있음

## all modality

통합한 데이터에서 train_test_split을 수행한 후 clinical, image의 feature로 각각 데이터셋 형성(저장위치: overlap 파일)
=> 크로스 모델시 각 데이터 셋의 인덱스를 통해서 concat이 된다

train_vt.py 사용

## train_vt.py 변경점

all_modality에서 clinical이랑 image만 사용하여 하도록 하였음 (snp 부분은 주석처리 및 함수 인자 제거를 통해 비활성화)

create_overlap_dataset에서 모든 파일을 pkl로 제공 -> 데이터 받는 부분을 read_pickle로 전환
(csv파일로도 변환해놨으니까 필요하면 사용)

make_img 함수 변경
