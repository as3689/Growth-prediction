import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # 데이터 로드 (저장된 파일명에 맞게 수정)
    X_test = np.load('data/X_test.npy')  # 테스트 이미지
    leaf_length_test = np.load('data/y_test_leaves.npy')  # 잎장
    leaf_width_test = np.load('data/y_test_weights.npy')  # 잎폭
    y_test_weights = np.load('data/y_test_weights.npy')  # 생체중

    # 모델 로드 시 mse와 mae를 직접 명시적으로 지정
    model = tf.keras.models.load_model(
        'models/crop_model.h5', 
        custom_objects={
            'MeanSquaredError': tf.keras.losses.MeanSquaredError(),  # mse
            'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError()  # mae
        }
    )

    # 평가 (생체중 예측을 평가)
    loss, mae = model.evaluate(
        [X_test, leaf_length_test, leaf_width_test], 
        y_test_weights,
        batch_size=32,
        verbose=1
    )
    
    # 결과 출력
    print(f"Test Loss: {loss}, Test MAE: {mae}")
