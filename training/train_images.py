import os       # 파일 경로를 다루거나, 환경 변수를 설정하는 등 운영체제와 관련된 기능을 사용
import random   # 무작위(random) 숫자나 데이터를 생성할 때 사용하는 파이썬 기본 라이브러리
import tensorflow as tf     #  딥러닝 프레임워크, as tf는 앞으로 코드에서 tensorflow 대신 tf라는 짧은 별명으로 부르겠다는 의미
from tensorflow import keras    # keras는 모델을 훨씬 쉽고 직관적으로 만들 수 있도록 도와주는 고수준 API Sequential 모델이나 Dense 같은 층(Layer)을 만들 때 사용
import numpy as np  # 행렬이나 벡터 같은 다차원 배열 연산 라이브러리
import pandas as pd # 표 형태 데이터 처리 라이브러리
import pickle5 as pickle    # 파이썬의 객체(데이터, 모델 등)를 파일 형태로 그대로 저장하거나 불러올 때 사용하는 라이브러리, pickle5를 pickle로 부른다.
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리로 matplotlib.pyplot을 plt로 부르기로 했다. 
from keras.models import Sequential # 신경망의 각 층(Layer)을 순서대로 차곡차곡 추가하여 모델 전체를 구성

# 정수 형태의 레이블(e.g., 0, 1, 2)을 원-핫 인코딩(One-Hot Encoding) 방식으로 변환해주는 유용한 함수
# 2라는 레이블을 3개의 클래스가 있는 문제에서 [0, 0, 1]과 같은 벡터 형태로 변환
from tensorflow.keras.utils import to_categorical

# 최적화 알고리즘(Optimizer) 중 하나, 오차를 가장 효과적으로 줄여나가는 방향으로 가중치(weights)를 업데이트하는 역할
from tensorflow.keras.optimizers import Adam

# 모델의 예측 성능을 종합적으로 보여주는 보고서를 생성 -> 정밀도(Precision), 재현율(Recall), F1-점수
from sklearn.metrics import classification_report

# 신경망 층(Layer) 블록들을 불러옵니다.
from keras.layers import Dense,Dropout,MaxPooling2D, Flatten, Conv2D

# 딥러닝 실험의 재현성을 위해 모든 무작위 시드를 고정하는 함수
def reset_random_seeds(seed):
    
    # 파이썬의 해시 함수 작동 방식을 제어하여 예측 가능하게 만듭니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 텐서플로우의 무작위 연산(예: 가중치 초기화) 시드를 고정합니다.
    tf.random.set_seed(seed)
    
    # 넘파이 라이브러리의 무작위 연산(예: 데이터 셔플링) 시드를 고정합니다.
    np.random.seed(seed)
    
    # 파이썬 내장 random 모듈의 시드를 고정합니다.
    random.seed(seed)


# 메임  함수
def main():
    
    # --- 1. 데이터 로딩 (Data Loading) ---
    # 이 섹션에서는 pickle 파일에 저장된 훈련/테스트용 이미지 데이터와 정답 레이블을 불러옵니다.
    
    # 훈련용 이미지 데이터(.pkl)를 바이너리 읽기 모드('rb')로 엽니다.
    with open("img_train.pkl", "rb") as fh:
        # pickle을 사용해 파일 내용을 data 변수에 로드합니다.
        data = pickle.load(fh)
    # 로드한 데이터를 Pandas DataFrame으로 변환하고, 'img_array' 컬럼(이미지 데이터)만 추출합니다.
    X_train_ = pd.DataFrame(data)["img_array"] 
    
    # 테스트용 이미지 데이터를 위와 동일한 방식으로 불러옵니다.
    with open("img_test.pkl", "rb") as fh:
        data = pickle.load(fh)
    X_test_ = pd.DataFrame(data)["img_array"]
    
    # 훈련용 정답 레이블 데이터를 불러옵니다.
    with open("img_y_train.pkl", "rb") as fh:
        data = pickle.load(fh)
    # DataFrame으로 변환 후 'label' 컬럼을 추출하고, 이를 NumPy 배열로 변환합니다.
    y_train = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    
    # 테스트용 정답 레이블 데이터를 위와 동일한 방식으로 불러옵니다.
    with open("img_y_test.pkl", "rb") as fh:
        data = pickle.load(fh)
    y_test = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    

    # --- 2. 데이터 전처리 (Data Preprocessing) ---
    # 이 섹션에서는 불러온 데이터의 형식을 모델 학습에 맞게 변환합니다.

    # 테스트 데이터의 레이블 값을 재조정합니다 (1 -> 2, 2 -> 1로 교체).
    # 임시값 -1을 사용한 3단계 교체 방식: (1) 2를 -1로 변경 -> (2) 1을 2로 변경 -> (3) -1(원래 2였던 값)을 1로 변경

    # 이유는 확실히는 모르겠으나 아마 기존 데이터 라벨링이 다르게 되었을거 같음.
    # ex) 레이블 0: 정상, 레이블 1: 주의, 레이블 2: 위험 가 반대로 되었다던지
    y_test[y_test == 2] = -1
    y_test[y_test == 1] = 2
    y_test[y_test == -1] = 1
    
    # 훈련 데이터의 레이블 값도 동일한 방식으로 재조정합니다.
    y_train[y_train == 2] = -1
    y_train[y_train == 1] = 2
    y_train[y_train == -1] = 1
    
    # Pandas Series에 담긴 이미지 배열들을 하나씩 꺼내 파이썬 리스트에 담을 준비를 합니다.
    X_train = []
    X_test = []
    
    # for문을 통해 X_train_의 각 이미지(Numpy 배열)를 X_train 리스트에 추가합니다.
    for i in range(len(X_train_)):
        X_train.append(X_train_.values[i])
        
    # X_test_에 대해서도 동일한 작업을 수행합니다.
    for i in range(len(X_test_)):
        X_test.append(X_test_.values[i])
    
    # 이미지 데이터가 담긴 파이썬 리스트를 Keras 모델이 입력으로 받을 수 있는 최종 Numpy 배열 형태로 변환합니다.
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    
    # --- 3. 실험 결과 저장을 위한 변수 초기화 ---
    # 여러 번의 실험 결과를 종합하기 위해 각 평가지표를 저장할 리스트를 생성합니다.
    acc = []
    f1 = []
    precision = []
    recall = []

    # 5번의 독립적인 실험을 수행하기 위해 1~199 사이에서 5개의 무작위 시드를 생성합니다.
    seeds = random.sample(range(1, 200), 5)


    # --- 4. 총 5회의 모델 훈련 및 평가 반복 ---
    # 생성된 5개의 시드를 하나씩 사용하며 전체 훈련/평가 과정을 5번 반복합니다.
    for seed in seeds:
        # 현재 루프의 시드(seed) 값으로 모든 라이브러리의 무작위성을 고정하여 실험의 재현성을 확보합니다.
        reset_random_seeds(seed)
        
        # --- 4.1. CNN 모델 구성 ---
        # Keras의 Sequential API를 사용해 CNN 모델을 정의합니다.
        model = Sequential()
        # 첫 번째 합성곱 층: 100개의 3x3 필터로 이미지의 특징(feature)을 추출합니다. 입력 이미지의 형태는 (72, 72, 3)입니다. -> 70*70*100 만든다.
        # relu는 활성화 함수 중 하나. Rectified Linear Unit으로 입력값이 0보다 작으면 0으로 만들고, 0보다 크면 입력값을 그대로 내보냅니다.
        model.add(Conv2D(100, (3, 3),  activation='relu', input_shape=(72, 72, 3)))
        # 맥스 풀링 층: 이미지의 크기를 절반으로 줄여(2x2) 중요한 특징만 강조하고 계산량을 줄입니다. -> 35*35*100 만든다.
        model.add(MaxPooling2D((2, 2)))
        # 드롭아웃 층: 훈련 중에 뉴런의 50%를 무작위로 비활성화하여 모델의 과적합(overfitting)을 방지합니다. (특징 맵(feature map)의 일부를 무작위로 0으로 만드는 방식)
        model.add(Dropout(0.5))
        # 두 번째 합성곱 층: 50개의 3x3 필터로 더 복잡한 특징을 추출합니다. -> 33*33*50으로 만든다.
        model.add(Conv2D(50, (3, 3), activation='relu'))
        # 두 번째 맥스 풀링 층 -> 16*16*50으로 만든다.
        model.add(MaxPooling2D((2, 2)))
        # 두 번째 드롭아웃 층 (비활성화 비율 30%)
        model.add(Dropout(0.3))
        # Flatten 층: 2D 형태의 특징 맵을 1D 벡터로 변환하여 Dense 층에 입력할 수 있도록 합니다. -> 12800*1로 만든다.
        model.add(Flatten())
        # 출력 층 (Dense): 3개의 뉴런으로 3개의 클래스에 대한 최종 확률을 계산합니다. softmax 활성화 함수를 사용합니다. 가중치행렬->12800*3
        model.add(Dense(3, activation = "softmax"))
        
        # --- 4.2. 모델 컴파일 및 학습 ---
        # 모델의 학습 방식을 설정합니다: 
        # Adam 옵티마이저 (내부 가중치(weights)를 어떻게 수정할지 결정) -> 옵티마이저는 이 손실 함수의 값을 최소화하는 방향으로 모델을 학습
        # sparse_categorical_crossentropy 손실 함수
        # sparse_categorical_accuracy 평가지표
        model.compile(Adam(learning_rate = 0.001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
        
        # 모델의 구조를 요약하여 출력합니다.
        model.summary()
        
        # 모델 훈련을 시작합니다. 전체 데이터를 50번 반복(epochs=50(반복횟수))하고, 
        # 한 번에 32개씩(batch_size=32(32개를 한 묶음으로)) 학습합니다.
        # 훈련 데이터의 10%는 검증(validation)용으로 사용하여 매 에포크마다 성능을 확인합니다.
        # verbose는 각 에포크마다 진행막대와 함께 현재까지의 성적을 상세히 보여달라는 의미
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,validation_split=0.1, verbose=1) 
        
        # --- 4.3. 모델 평가 및 성능 측정 ---
        # 훈련된 모델을 테스트 데이터로 평가하여 최종 손실(loss)과 정확도(accuracy)를 계산합니다.
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        # 이번 실험의 정확도를 acc 리스트에 추가합니다.
        acc.append(score[1])
        
        # 테스트 데이터에 대한 모델의 예측 확률값(softmax 결과)을 계산합니다.
        test_predictions = model.predict(X_test)
        # sklearn의 classification_report를 사용하기 위해 정답 레이블을 원-핫 인코딩 형식으로 변환(to_categorical)합니다.
        # 원-핫 인코딩은 여러 선택지 중 1개를 1로 하고 나머지를 0으로 하는 데이터 표현 방식
        test_label = to_categorical(y_test,3)

        # 원-핫 인코딩된 정답 레이블에서 가장 큰 값의 인덱스를 찾아 다시 정수(0, 1, 2) 형태로 변환합니다.
        # np.argmax는 배열에서 가장 큰 값의 위치(인덱스)를 찾아주는 함수입니다. 
        # axis=1은 이 작업을 각 행(row)별로 독립적으로 수행하라는 명령어입니다.
        true_label= np.argmax(test_label, axis =1)

        # 모델이 예측한 확률값 배열에서 가장 큰 값의 인덱스를 찾아 최종 예측 레이블(0, 1, 2)을 결정합니다.
        predicted_label= np.argmax(test_predictions, axis =1)
        
        # 실제 정답(true_label)과 모델의 예측(predicted_label)을 비교하여 상세 성능 보고서를 생성합니다.
        cr = classification_report(true_label, predicted_label, output_dict=True)
        # 보고서에서 'macro avg'(전체 클래스의 평균)의 정밀도, 재현율, F1 점수를 각각의 리스트에 추가합니다.
        precision.append(cr["macro avg"]["precision"])
        recall.append(cr["macro avg"]["recall"])
        f1.append(cr["macro avg"]["f1-score"])
    
    # --- 5. 최종 결과 종합 및 출력 ---
    # 5번의 실험이 모두 끝난 후, 저장된 평가지표들의 평균과 표준편차를 계산하여 출력합니다.
    
    # 5번의 실험에서 얻은 정확도의 '평균'을 계산하여 모델의 일반적인 성능을 확인합니다.
    print("Avg accuracy: " + str(np.array(acc).mean()))
    print("Avg precision: " + str(np.array(precision).mean()))
    print("Avg recall: " + str(np.array(recall).mean()))
    print("Avg f1: " + str(np.array(f1).mean()))
    
    # 5번의 실험에서 얻은 정확도의 '표준편차'를 계산하여 성능이 얼마나 안정적인지(낮을수록 안정적) 확인합니다.
    print("Std accuracy: " + str(np.array(acc).std()))
    print("Std precision: " + str(np.array(precision).std()))
    print("Std recall: " + str(np.array(recall).std()))
    print("Std f1: " + str(np.array(f1).std()))
    
    # 각 실험별 개별 점수를 확인하기 위해 리스트 전체를 출력합니다.
    print(acc)
    print(precision)
    print(recall)
    print(f1)
    
# 이 스크립트 파일이 직접 실행될 때만 main() 함수를 호출합니다.
if __name__ == '__main__':
    main()