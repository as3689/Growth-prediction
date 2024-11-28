import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model():
    """CNN 모델 설계"""
    input_image = layers.Input(shape=(224, 224, 3))  # 이미지 입력
    input_length = layers.Input(shape=(1,))  # 잎장 입력
    input_width = layers.Input(shape=(1,))  # 잎폭 입력

    # CNN 네트워크
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_image)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # CNN 결과와 다른 입력 데이터를 결합
    combined = layers.Concatenate()([x, input_length, input_width])
    dense = layers.Dense(128, activation='relu')(combined)
    output = layers.Dense(1)(dense)  # 예측할 값 (생체중)

    model = models.Model(inputs=[input_image, input_length, input_width], outputs=output)
    return model

if __name__ == '__main__':
    # 데이터 로드
    X_train = np.load('data/X_train.npy')  # 훈련 이미지
    X_test = np.load('data/X_test.npy')    # 테스트 이미지
    
    # 잎장, 잎폭, 생체중 데이터 로드
    leaf_length_train = np.load('data/y_train_leaves.npy')  # 잎장
    leaf_width_train = np.load('data/y_train_weights.npy')  # 잎폭
    y_train_weights = np.load('data/y_train_weights.npy')  # 생체중

    # 모델 생성 및 컴파일
    model = create_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 손실 함수 MSE, 평가 지표 MAE

    # 모델 훈련
    model.fit(
        [X_train, leaf_length_train, leaf_width_train],  # 입력 데이터 (이미지, 잎장, 잎폭)
        y_train_weights,  # 훈련 라벨 (생체중)
        epochs=10,
        batch_size=32,
        validation_split=0.2  # 20%를 검증 데이터로 사용
    )

    # 모델 저장
    model.save('models/crop_model.h5')

    print("모델 훈련 완료 및 저장!")
