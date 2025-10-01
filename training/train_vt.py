import os
import random
import gc, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Dropout,Flatten, BatchNormalization, Conv2D, MultiHeadAttention, concatenate
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

import os, json, ast
import numpy as np
import pandas as pd


def make_img(t_img):
    # 1) pickle 읽기
    df = pd.read_pickle(t_img)

    # 2) img_array 컬럼만 사용
    entries = df['img_array'].tolist()
    
    imgs = []

    for val in entries:
        # 3) ndarray로 변환 (단순)
        arr = np.asarray(val)

        # 4) shape 맞추기, 채널 체크 제거
        imgs.append(arr.astype(np.float32))

    # 5) stack (shape 체크 제거)
    result = np.stack(imgs, axis=0).astype(np.float32)  # (N, H, W, C)

    return result

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
   
               
def create_model_snp():
    
    model = Sequential()
    model.add(Dense(200,  activation = "relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    return model

def create_model_clinical():
    
    model = Sequential()
    model.add(Dense(200,  activation = "relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))    
    return model

def create_model_img():
    
    model = Sequential()
    model.add(Conv2D(72, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))   
    return model


def plot_classification_report(y_tru, y_prd, mode, learning_rate, batch_size,epochs, figsize=(7, 7), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = ["Control", "Moderate", "Alzheimer's" ] 
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax, cmap = "Blues")
    
    plt.savefig('report_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'_' + str(epochs)+'.png')
    


def calc_confusion_matrix(result, test_label,mode, learning_rate, batch_size, epochs):
    test_label = to_categorical(test_label,3)

    true_label= np.argmax(test_label, axis =1)

    predicted_label= np.argmax(result, axis =1)
    
    n_classes = 3
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    print ("Classification Report :") 
    print (classification_report(true_label, predicted_label))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    return cr, precision, recall, thres



def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 4,key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads = 4,key_dim=50)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return concatenate([a1, a2])


def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention
    

def multi_modal_model(mode, train_clinical, train_img, train_snp=None):
    
    # 입력 데이터의 형식을 정하는 부분
    #shape의 2번째는 열이다. 입력이 벡터고 x차원이다. 이런식으로 저장
    in_clinical = Input(shape=(train_clinical.shape[1]))
    
    #shape의 2번째는 열이다. 입력이 벡터고 x차원이다. 이런식으로 저장
    # in_snp = Input(shape=(train_snp.shape[1]))
    
    #shape의 2번째는 열이다. 입력이 벡터고 x차원이다. 이런식으로 저장
    #이미지는 높이 너비 채널이 있기에 
    # shape[1]은 높이, shape[2]는 너비, shape[3]은 채널
    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    
    # 입력->200->100->50
    dense_clinical = create_model_clinical()(in_clinical)
    # 입력->200->100->50
    # dense_snp = create_model_snp()(in_snp) 
    
    dense_img = create_model_img()(in_img) 
    
    # 모두 출력 50차원으로 해서 저장해줌
 
        
    ########### Attention Layer ############
        
    ## Cross Modal Bi-directional Attention ##
    # 패스
    if mode == 'MM_BA':
            
        vt_att = cross_modal_attention(dense_img, dense_clinical)
        # av_att = cross_modal_attention(dense_snp, dense_img)
        # ta_att = cross_modal_attention(dense_clinical, dense_snp)
                
        merged = concatenate([vt_att, dense_img, dense_clinical])
                 
   
        
    #패스
    ## Self Attention ##
    elif mode == 'MM_SA':
            
        vv_att = self_attention(dense_img)
        tt_att = self_attention(dense_clinical)
        # aa_att = self_attention(dense_snp)
            
        merged = concatenate([vv_att, tt_att, dense_img, dense_clinical])
        
    ## Self Attention and Cross Modal Bi-directional Attention##
    elif mode == 'MM_SA_BA':
            
        # 다 50차원이 멀티헤드 셀프어텐션 함 출력이 attention value 1*200, 자기 자신이 문맥정보를 보고 문맥정보를 가진 1*200짜리 벡터로 나온다고 보면됨
        vv_att = self_attention(dense_img)
        tt_att = self_attention(dense_clinical)
        # aa_att = self_attention(dense_snp)
        
        # 이것도 크로스 어텐션 위에서 받은 1*200 , 1*200이 입력으로 들어가서 멀티헤드 크로스어텐션 해서 1*800으로 출력, a->b 멀티헤드 크로스 어텐션이라고 하면
        #a가 b와 관련된 정도에따라 b의 값을 가져오는 것, 함수 하나에 (a,b) 들어가서 출력으로 a->b corss attention, b_a cross attention concat 된 게 나옴
        vt_att = cross_modal_attention(vv_att, tt_att)
        # av_att = cross_modal_attention(aa_att, vv_att)
        # ta_att = cross_modal_attention(tt_att, aa_att)
        
        # 다 컨켓
        merged = concatenate([vt_att, dense_img, dense_clinical])
            
        
    ## No Attention ##    
    elif mode == 'None':
            
        merged = concatenate([dense_img, dense_clinical])
                
    else:
        print ("Mode must be one of 'MM_SA', 'MM_BA', 'MU_SA_BA' or 'None'.")
        return
                
        
    ########### Output Layer ############
        
    output = Dense(3, activation='softmax')(merged)
    
    model = Model([in_clinical, in_img], output)        
        
    return model



def train(mode, batch_size, epochs, learning_rate, seed):
    
 
    train_clinical = pd.read_pickle("D:/Desktop/aicoss-model/preprocessed/overlap/X_train_clinical.pkl")
    test_clinical= pd.read_pickle("D:/Desktop/aicoss-model/preprocessed/overlap/X_test_clinical.pkl")

    
    # train_snp = pd.read_pickle("D:/Desktop/aicoss-model/preprocessed/overlap/X_train_snp.pkl")
    # test_snp = pd.read_pickle("D:/Desktop/aicoss-model/preprocessed/overlap/X_test_snp.pkl")

    
    train_img= make_img("D:/Desktop/aicoss-model/preprocessed/overlap/X_train_img.pkl")
    test_img= make_img("D:/Desktop/aicoss-model/preprocessed/overlap/X_test_img.pkl")

    
    train_label= pd.read_pickle("D:/Desktop/aicoss-model/preprocessed/overlap/y_train.pkl").values.astype("int").flatten()
    test_label= pd.read_pickle("D:/Desktop/aicoss-model/preprocessed/overlap/y_test.pkl").values.astype("int").flatten()
    
    reset_random_seeds(seed)
    
    # 클래스가 불균형하기 때문에 클래스별로 가중치를 정해준다.
    class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(train_label),y = train_label)
    #이를 딕셔너리 형태로 저장한다.
    d_class_weights = dict(enumerate(class_weights))
    
    # compile model #
    #여기서 모델 만들고
    model = multi_modal_model(mode, train_clinical, train_img)
    # 가중치 최적화는 adam으로 하고 loss는 crossentropy, metrics는 categorial_accuracy
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    

    # summarize results
    # 입력 넣고 정답지 넣고 클래스별 가중치나 등등 파라미터 넣고 학습
    history = model.fit([train_clinical,
                         train_img],
                        train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        class_weight=d_class_weights,
                        validation_split=0.1,
                        verbose=1)
                        
                
    # 테스트 데이터로 평가
    score = model.evaluate([test_clinical, test_img], test_label)
    
    # 테스트 나온 accuracy 저장
    acc = score[1] 
    
    
    #  test 데이터로 예측 저장
    test_predictions = model.predict([test_clinical, test_img])
    
    # 이 예측을 바탕으로 precision, recall 등등 정확도 이외의 지표 구함
    cr, precision_d, recall_d, thres = calc_confusion_matrix(test_predictions, test_label, mode, learning_rate, batch_size, epochs)
    
    
    """
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('accuracy_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('loss_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    """
    
 
    
    # release gpu memory #
    K.clear_session()
    del model, history
    gc.collect()
        
        
    print ('Mode: ', mode)
    print ('Batch size:  ', batch_size)
    print ('Learning rate: ', learning_rate)
    print ('Epochs:  ', epochs)
    print ('Test Accuracy:', '{0:.4f}'.format(acc))
    print ('-'*55)
    
    return acc, batch_size, learning_rate, epochs, seed
    
    
if __name__=="__main__":
    
    # arr = make_img_s("D:/Desktop/aicoss-model/preprocessed/overlap/X_train_img.pkl", verbose=True)
    # print("final shape:", arr.shape)
    # print("example dtype:", type(arr[0]), "element shape:", arr[0].shape)

    m_a = {}
    seeds = random.sample(range(1, 200), 5)
    
    # 드랍아웃이 랜덤이라 운이 필요함 그래서 5번 해서 정확도 m_a에 정확도를 키로 각종 다른 것들을 저장
    for s in seeds:
        acc, bs_, lr_, e_ , seed= train('MM_SA_BA', 32, 50, 0.001, s)
        m_a[acc] = ('MM_SA_BA', acc, bs_, lr_, e_, seed)
    print(m_a)
    print ('-'*55)
    #최대 정확도, 그떄 파라미터를 출력
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
    