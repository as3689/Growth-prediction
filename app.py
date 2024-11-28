from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

# Flask 앱 초기화
app = Flask(__name__)

# 학습된 모델 로드
model = tf.keras.models.load_model('models/crop_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    """생체중 예측 API"""
    try:
        # 요청에서 이미지 파일과 파라미터 수신
        file = request.files['image']  # 업로드된 이미지 파일
        leaf_length = float(request.form['leaf_length'])  # 잎장
        leaf_width = float(request.form['leaf_width'])    # 잎폭

        # 이미지 전처리: 파일을 읽고 리사이즈
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))  # 모델에 맞는 입력 크기로 조정
        img = np.expand_dims(img, axis=0)  # 배치 차원 추가

        # 모델 예측
        prediction = model.predict([img, np.array([[leaf_length]]), np.array([[leaf_width]])])

        # 결과 반환
        return jsonify({'predicted_weight': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000)
