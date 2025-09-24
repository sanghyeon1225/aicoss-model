
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential


# 랜덤 시드 고정 (재현성을 보장하기 위함)
def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def main():
    
        #this is created in the genetic preprocess jupyter notebook
        X_train = pd.read_pickle("X_train_vcf.pkl")
        y_train = pd.read_pickle("y_train_vcf.pkl")

        X_test = pd.read_pickle("X_test_vcf.pkl")
        y_test = pd.read_pickle("y_test_vcf.pkl")


        acc = []
        f1 = []
        precision = []
        recall = []
        seeds = random.sample(range(1, 200), 5)

        for seed in seeds:
            reset_random_seeds(seed)
            
            # 모델 구조 구현
            model = Sequential()
            
            # 입력층 / 15965개의 특성을 가진 데이터를 입력으로 하겠다
            # 유전체 데이터의 경우 SNP(단일 염기 다향성)등의 유전 변이 개수에 해당함
            model.add(Dense(128, input_shape = (15965,), activation = "relu")) 
            
            # Dropout을 통해 과적합을 방지함 (훈련 과정 중 50% or 30% 만큼의 뉴런을
            # 랜덤하게 비활성화시켜 모델이 훈련 데이터에만 의존하지 않도록 함)
            model.add(Dropout(0.5))
            
            # 은닉층 1
            model.add(Dense(64, activation = "relu"))
            model.add(Dropout(0.5))

            # 은닉층 2
            model.add(Dense(32, activation = "relu"))
            model.add(Dropout(0.3))
            
            # 은닉층 3
            model.add(Dense(32, activation = "relu"))
            model.add(Dropout(0.3))

            # 출력층 (3개의 클래스로 분류라서 뉴런이 3개임)
            # softmax를 사용해서 출력값을 각 클래스에 속할 확률로 변환함
            # 예 : [0.1, 0.7, 0.2]
            model.add(Dense(3, activation = "softmax"))
            
            # 은닉층1 (128개 뉴런) -> 은닉층2 (64개 뉴런) -> 은닉층3 (32개 뉴런) -> 은닉층4 (32개 뉴런) -> 출력층
            
            # 모델 컴파일 과정
            # 학습률: 0.001, sparse_categorical_crossentropy: 정답이 [0, 1, 2]와 같이 정수 형태일 때 사용하는 손실 함수
            # metrics = ["sparse_categorical_accuracy"]: 평가 지표 
            model.compile(Adam(learning_rate = 0.001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
            
            # 모델 학습 (50번의 에폭 반복 학습 마다, 10%의 데이터로 검증)
            history = model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split = 0.1, verbose=1) 
            
            # evaluate를 통해 모델의 정확도를 분석
            score = model.evaluate(X_test, y_test, verbose=0)
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
            acc.append(score[1])
            
            # predict를 통해 모델의 분류 수행
            # test_predictions = [
            #    [0.1, 0.8, 0.1],  # 첫 번째 데이터: 클래스 1일 확률이 80%로 가장 높음
            #    [0.9, 0.05, 0.05], # 두 번째 데이터: 클래스 0일 확률이 90%로 가장 높음
            #    [0.2, 0.3, 0.5],  # 세 번째 데이터: 클래스 2일 확률이 50%로 가장 높음
            #    ]
            test_predictions = model.predict(X_test)
            
            # y_test는 [1, 0, 2, ...] 이런 형식
            # to_categorical을 통해 [[0,1,0], [1,0,0], [0,0,1], ...] 이런식으로 원-핫 벡터로 변환
            test_label = to_categorical(y_test,3)
            
            # argmax를 통해 다시 원래 정수 라벨로 되돌림 (사실 위 두 코드는 안해도 되는데 아래 predicted_label과 코드 통일성을 위해서 한듯)
            true_label= np.argmax(test_label, axis =1)
            
            # 그냥 사실 true_label = y_test 이런식으로 한줄만 해도 됨 !!
            
            
            # 모델이 예측한 확률 중 가장 높은 인덱스 뽑아내기
            # [0.1, 0.8, 0.1]이라면 1을 선택
            # predicted_label = [1, 0, 2, ...] 1차원 정수 배열 형태로 저장
            predicted_label= np.argmax(test_predictions, axis =1)
            
            # 상세 지표를 확인하기 위한 report 생성 (true_label, predicted_label 사용)
            cr = classification_report(true_label, predicted_label, output_dict=True)
            
            # report(cr)에서 precision, recall, f1 점수를 뽑아와서 저장
            precision.append(cr["macro avg"]["precision"])
            recall.append(cr["macro avg"]["recall"])
            f1.append(cr["macro avg"]["f1-score"])
            
            
        # 결과 출력 !!
        print("Avg accuracy: " + str(np.array(acc).mean()))
        print("Avg precision: " + str(np.array(precision).mean()))
        print("Avg recall: " + str(np.array(recall).mean()))
        print("Avg f1: " + str(np.array(f1).mean()))
        print("Std accuracy: " + str(np.array(acc).std()))
        print("Std precision: " + str(np.array(precision).std()))
        print("Std recall: " + str(np.array(recall).std()))
        print("Std f1: " + str(np.array(f1).std()))
        print(acc)
        print(precision)
        print(recall)
        print(f1)

    

if __name__ == '__main__':
    main()
    
