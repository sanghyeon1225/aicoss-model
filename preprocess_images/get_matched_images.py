import pandas as pd
from pathlib import Path

# 파일 경로 바꾸기
clinical_fp = "D:/Desktop/aicoss-model/preprocessed/clinical.csv"
mri_meta_fp = "D:/Desktop/aicoss-model/mri_meta.csv"
out_matched_fp = "D:/Desktop/aicoss-model/preprocessed/mri_meta_labeled.csv"

# 로드
clin = pd.read_csv(clinical_fp, dtype=str)   # string으로 읽으면 매칭 편함
mmeta = pd.read_csv(mri_meta_fp, dtype=str)

# 컬럼 예시: clin: PTID, VISCODE, VISCODE2, DX (또는 DXSUM의 진단 컬럼)
#            mmeta: subject_id, image_visit, [image identifiers... e.g., image_id, series_id, loni_image_id, file_name]
# 필요시 실제 컬럼명을 여기서 확인/교체하세요
print("clinical columns:", clin.columns.tolist())
print("mri_meta columns:", mmeta.columns.tolist())

# 전처리 팁: 공백/대소문자 정리 (VISCODE 계열은 문자열 정확히 일치시켜야 함)
clin['VISCODE'] = clin['VISCODE'].str.strip().fillna("")
clin['VISCODE2'] = clin['VISCODE2'].str.strip().fillna("")
clin['PTID'] = clin['PTID'].str.strip()
mmeta['image_visit'] = mmeta['image_visit'].str.strip().fillna("")
mmeta['subject_id'] = mmeta['subject_id'].str.strip()

# 우리는 두 방법으로 매칭: 1) PTID == subject_id (필수), 2) image_visit == VISCODE or VISCODE2
# 방법: clinical을 기준으로 각 행마다 해당 subject/visit에 해당하는 모든 mri_meta row를 연결

# (1) clinical의 PTID 기준으로 mri_meta 후보 선별 (inner join on PTID/subject_id)
joined = clin.merge(mmeta, left_on='PTID', right_on='subject_id', how='left', suffixes=('_clin','_mri'))

# (2) visit 코드 일치 여부 필터
# visit이 일치하는 행을 살린다 (VISCODE or VISCODE2 matches image_visit)
mask_visit = (
    (joined['image_visit'] == joined['VISCODE']) |
    (joined['image_visit'] == joined['VISCODE2'])
)

matched = joined[mask_visit].copy()

# matched에 라벨 컬럼 추가
matched = matched.rename(columns={'DX':'Group'}) 
# 필요한 경우 diagnosis 값 정리
matched['Group'] = matched['Group'].fillna('Unknown')

# 결과 확인
print(f"Total clinical rows: {len(clin)}")
print(f"Total mri_meta rows: {len(mmeta)}")
print(f"Matched rows: {len(matched)}")

# 한 이미지(row)가 여러 clinical row와 매칭되는지(중복 라벨) 확인
dup_images = matched.duplicated(subset=['image_id'], keep=False)

# 저장: IDA에서 검색/다운로드할 때 쓸 식별자 열들(예: series_id, image_id, file_name 등)을 포함해서 저장
cols_to_save = ['subject_id','image_visit','image_id','Group']
matched[cols_to_save].to_csv(out_matched_fp, index=False)
print(f"Saved matched & labeled metadata to: {out_matched_fp}")