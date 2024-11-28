import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_data(image_dir, label_file):
    """
    이미지를 로드하고 라벨 데이터를 함께 불러옵니다.
    - image_dir: 이미지 폴더 경로
    - label_file: 라벨 CSV 파일 경로
    """
    labels = pd.read_csv(label_file)  # 라벨 데이터 로드
    images, weights, leaves, leafareas = [], [], [], []

    for _, row in labels.iterrows():
        img_name = row['images']  # 'images' 컬럼에서 파일명 가져오기
        img_path = os.path.join(image_dir, img_name)  # 이미지 경로 결합

        if os.path.exists(img_path):  # 이미지 파일이 존재하는 경우
            img = cv2.imread(img_path)  # 이미지 읽기
            img = cv2.resize(img, (224, 224))  # 이미지 크기 조정
            images.append(img)
            weights.append(row['weight'])  # 'weight' 값 추가
            leaves.append(row['leaves'])  # 'leaves' 값 추가
            leafareas.append(row['leafarea'])  # 'leafarea' 값 추가
        else:
            print(f"이미지 파일을 찾을 수 없음: {img_path}")  # 없는 파일 경고

    return np.array(images), np.array(weights), np.array(leaves), np.array(leafareas)

def split_data(images, weights, leaves, leafareas):
    """
    Train-Test 데이터를 80:20 비율로 분리합니다.
    - images: 이미지 데이터
    - weights, leaves, leafareas: 라벨 데이터
    """
    return train_test_split(images, weights, leaves, leafareas, test_size=0.2, random_state=42)

if __name__ == '__main__':
    # 데이터 경로 설정
    image_dir = 'data/images/images'  # 이미지 디렉토리
    label_file = 'data/labels.csv'  # 라벨 CSV 파일

    # 데이터 로드
    images, weights, leaves, leafareas = load_data(image_dir, label_file)

    # 데이터 분리
    X_train, X_test, y_train_weights, y_test_weights, y_train_leaves, y_test_leaves, y_train_leafareas, y_test_leafareas = split_data(
        images, weights, leaves, leafareas
    )

    # 데이터 저장
    np.save('data/X_train.npy', X_train)  # 훈련 이미지
    np.save('data/X_test.npy', X_test)  # 테스트 이미지
    np.save('data/y_train_weights.npy', y_train_weights)  # 훈련 weight
    np.save('data/y_test_weights.npy', y_test_weights)  # 테스트 weight
    np.save('data/y_train_leaves.npy', y_train_leaves)  # 훈련 leaves
    np.save('data/y_test_leaves.npy', y_test_leaves)  # 테스트 leaves
    np.save('data/y_train_leafareas.npy', y_train_leafareas)  # 훈련 leafarea
    np.save('data/y_test_leafareas.npy', y_test_leafareas)  # 테스트 leafarea

    print("데이터 저장 완료!")
