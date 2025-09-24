# 알츠하이머병 분류를 위한 임상 데이터 기반 단일 모달리티 딥러닝 모델
# 입력: 185차원 임상 피처 벡터 (인구통계, 신경학적 검사, 인지 평가 데이터)
# 출력: 3클래스 분류 (0: Control, 1: MCI, 2: AD)

import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential



def reset_random_seeds(seed):
    """
    재현 가능한 실험을 위한 모든 랜덤 시드 고정
    Args:
        seed (int): 시드 값
    """
    os.environ['PYTHONHASHSEED']=str(seed)  # 파이썬 해시 시드 고정
    tf.random.set_seed(seed)                # TensorFlow 랜덤 시드 고정
    np.random.seed(seed)                    # NumPy 랜덤 시드 고정
    random.seed(seed)                       # Python 내장 random 시드 고정


def main():
    """
    임상 데이터 기반 알츠하이머병 분류 모델의 훈련 및 평가
    5-fold 교차검증으로 모델 성능의 안정성 측정
    """
    # 전처리된 임상 데이터 로드 (preprocess_clinical/create_clinical_dataset.ipynb에서 생성)
    X_train = pd.read_pickle("X_train_c.pkl")  # 훈련용 임상 피처 (n_samples, 185)
    y_train = pd.read_pickle("y_train_c.pkl")  # 훈련용 라벨 (n_samples,) [0, 1, 2]

    X_test = pd.read_pickle("X_test_c.pkl")    # 테스트용 임상 피처
    y_test = pd.read_pickle("y_test_c.pkl")    # 테스트용 라벨

    # 5-fold 교차검증을 위한 성능 지표 저장 리스트
    acc = []        # 정확도 (Accuracy) 저장
    f1 = []         # F1-스코어 저장
    precision = []  # 정밀도 (Precision) 저장
    recall = []     # 재현율 (Recall) 저장

    # 5개의 서로 다른 랜덤 시드로 모델 성능의 분산 측정
    seeds = random.sample(range(1, 200), 5)  # 1-200 범위에서 5개 시드 선택

    for seed in seeds:  # 각 시드별로 모델 훈련 및 평가
        reset_random_seeds(seed)  # 현재 실험의 재현성을 위한 시드 고정

        # Sequential 모델 생성 (순차적 레이어 구조)
        model = Sequential()

        # 첫 번째 은닉층: 185 → 128 (임상 피처 압축)
        model.add(Dense(128, input_shape = (185,), activation = "relu"))
        model.add(BatchNormalization())  # 내부 공변량 변화 감소로 훈련 안정화
        model.add(Dropout(0.5))         # 50% 뉴런을 랜덤하게 비활성화 (과적합 방지)

        # 두 번째 은닉층: 128 → 64 (특징 추상화)
        model.add(Dense(64, activation = "relu"))
        model.add(BatchNormalization())  # 배치 정규화로 기울기 소실 방지
        model.add(Dropout(0.3))         # 30% 드롭아웃 (첫 번째보다 완화)

        # 세 번째 은닉층: 64 → 50 (고수준 특징 학습)
        model.add(Dense(50, activation = "relu"))
        model.add(BatchNormalization())  # 훈련 안정성 증대
        model.add(Dropout(0.2))         # 20% 드롭아웃 (가장 완화된 정규화)

        # 출력층: 50 → 3 (Control, MCI, AD 분류)
        model.add(Dense(3, activation = "softmax"))  # 3클래스 확률 분포 출력
        
        # 모델 컴파일 (최적화 알고리즘, 손실함수, 평가지표 설정)
        model.compile(
            optimizer=Adam(learning_rate = 0.0001),      # Adam 최적화기 (낮은 학습률로 안정적 학습)
            loss="sparse_categorical_crossentropy",      # 정수 라벨용 다중분류 손실함수
            metrics = ["sparse_categorical_accuracy"]    # 분류 정확도 모니터링
        )

        model.summary()  # 모델 구조 및 파라미터 수 출력

        # 모델 훈련 (Early Stopping 없이 고정 에포크)
        history = model.fit(
            X_train, y_train,      # 훈련 데이터
            epochs=100,            # 100 에포크 훈련
            validation_split=0.1,  # 훈련 데이터의 10%를 검증용으로 분할
            batch_size=32,         # 미니배치 크기
            verbose=1              # 훈련 과정 출력
        ) 

        # 테스트 데이터로 모델 성능 평가
        score = model.evaluate(X_test, y_test, verbose=0)  # verbose=0: 진행상황 출력 안함
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        acc.append(score[1])  # 현재 fold의 정확도 저장

        # 테스트 데이터 예측 및 성능 지표 계산
        test_predictions = model.predict(X_test)  # 각 클래스별 확률 예측 (shape: [n_samples, 3])

        # 라벨 형태 통일을 위한 변환
        test_label = to_categorical(y_test, 3)    # 정수 라벨 → 원-핫 인코딩 ([0,1,2] → [[1,0,0],[0,1,0],[0,0,1]])
        true_label = np.argmax(test_label, axis=1)        # 원-핫 → 정수 라벨 (역변환)
        predicted_label = np.argmax(test_predictions, axis=1)  # 확률 → 예측 클래스

        # sklearn을 사용한 상세 분류 성능 평가
        cr = classification_report(true_label, predicted_label, output_dict=True)

        # 각 클래스의 성능을 동등하게 평가하는 Macro Average 사용
        precision.append(cr["macro avg"]["precision"])  # 정밀도 (과진단 방지)
        recall.append(cr["macro avg"]["recall"])        # 재현율 (누락 방지)
        f1.append(cr["macro avg"]["f1-score"])          # F1-스코어 (정밀도와 재현율의 조화평균)
    
    # 5-fold 교차검증 결과 통계 출력
    print("\n=== 5-Fold 교차검증 결과 통계 ===")

    # 평균 성능 지표 (모델의 전반적 성능)
    print("평균 성능 지표:")
    print("Avg accuracy: " + str(np.array(acc).mean()))          # 전체 정확도 평균
    print("Avg precision: " + str(np.array(precision).mean()))   # 정밀도 평균 (과진단 방지)
    print("Avg recall: " + str(np.array(recall).mean()))         # 재현율 평균 (누락 방지)
    print("Avg f1: " + str(np.array(f1).mean()))                # F1-스코어 평균 (균형 지표)

    # 표준편차 (모델 성능의 안정성/일관성 측정)
    print("\n성능 안정성 (표준편차, 낮을수록 안정적):")
    print("Std accuracy: " + str(np.array(acc).std()))          # 정확도 분산
    print("Std precision: " + str(np.array(precision).std()))   # 정밀도 분산
    print("Std recall: " + str(np.array(recall).std()))         # 재현율 분산
    print("Std f1: " + str(np.array(f1).std()))                # F1-스코어 분산

    # 각 fold별 상세 결과 출력 (디버깅 및 분석용)
    print("\n=== Fold별 상세 결과 ===")
    print("Accuracy per fold:", acc)
    print("Precision per fold:", precision)
    print("Recall per fold:", recall)
    print("F1-score per fold:", f1)
    
    
    """
    # 훈련 과정 시각화 (선택적 사용)
    # 주의: 현재는 마지막 fold의 history만 시각화됨

    # 정확도 변화 그래프
    plt.plot(history.history['sparse_categorical_accuracy'])    # 훈련 정확도
    plt.plot(history.history['val_sparse_categorical_accuracy']) # 검증 정확도
    plt.title('Model Accuracy Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    plt.clf()  # 그래프 초기화

    # 손실 함수 변화 그래프
    plt.plot(history.history['loss'])         # 훈련 손실
    plt.plot(history.history['val_loss'])     # 검증 손실
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.savefig('clinical_loss.png')  # 그래프 저장 (선택사항)
    plt.show()
    """


if __name__ == '__main__':
    """
    메인 실행부
    - 스크립트가 직접 실행될 때만 main() 함수 호출
    - 다른 모듈에서 import할 때는 실행되지 않음
    """
    main()

