import cv2
import numpy as np
import os
import re

# 기본 설정
INPUT_DIR = 'data/images_multi'  # 원본 이미지가 저장된 디렉토리
OUTPUT_DIR = 'data/images'       # 처리된 작물 저장 경로
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 출력 디렉토리 생성


def detect_and_segment_crops(input_image_path, output_dir, base_name):
    # 이미지 불러오기
    image = cv2.imread(input_image_path)
    original = image.copy()

    # 이미지 색상 범위 설정 (녹색 범위)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  # 녹색의 최소 값 (Hue, Saturation, Value)
    upper_green = np.array([85, 255, 255])  # 녹색의 최대 값
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)  # 녹색만 마스크로 추출

    # 녹색 부분만 추출
    green_crops = cv2.bitwise_and(image, image, mask=green_mask)

    # 마스크된 이미지를 확인하여 녹색 부분만 남은지 확인
    cv2.imwrite("green_crops.png", green_crops)  # 디버깅용으로 저장

    # 전처리: 녹색 부분만을 잘라내기 위한 이진화 및 컨투어 검출
    gray = cv2.cvtColor(green_crops, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 컨투어 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 감지된 작물 개수 확인
    print(f"이미지 {base_name}: 감지된 녹색 작물 개수 {len(contours)}")

    for i, contour in enumerate(contours):
        # 바운딩 박스 생성
        x, y, w, h = cv2.boundingRect(contour)
        
        # 너무 작은 영역은 무시
        if w < 10 or h < 10:
            continue

        crop = original[y:y+h, x:x+w]  # 작물 영역 잘라내기

        # 배경 제거(누끼 따기)
        crop_nobg = remove_background(crop)

        # 결과 파일 이름 생성 (날짜_작물번호.png 형식)
        output_file_name = f"{base_name}_{i+1}.png"
        output_path = os.path.join(output_dir, output_file_name)

        # 결과 저장
        cv2.imwrite(output_path, crop_nobg)
        print(f"작물 {i+1} 저장 완료: {output_path}")


def remove_background(crop):
    # GrabCut 초기 마스크 생성
    mask = np.zeros(crop.shape[:2], np.uint8)

    # GrabCut 알고리즘을 위한 임시 모델
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # ROI(작물 전체 영역) 설정
    h, w = crop.shape[:2]
    # 이미지가 너무 작은 경우, rect 영역을 더 적합하게 설정
    rect = (5, 5, w-10, h-10) if w > 10 and h > 10 else (0, 0, w, h)

    # GrabCut 실행
    try:
        cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"GrabCut 오류: {e}")
        return crop  # 오류가 발생하면 원본 이미지를 반환

    # 배경 제거된 마스크 생성
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 배경 제거한 이미지 생성
    crop_nobg = crop * mask2[:, :, np.newaxis]

    # 배경을 투명하게 설정 (RGBA 포맷)
    b, g, r = cv2.split(crop_nobg)
    alpha = (mask2 * 255).astype('uint8')  # 투명도 채널 생성
    crop_nobg_rgba = cv2.merge((b, g, r, alpha))  # RGBA로 합치기

    return crop_nobg_rgba


if __name__ == '__main__':
    # 디렉토리 내 모든 이미지 파일 처리
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith(('.jpg', '.png')):  # 지원하는 이미지 파일 확장자
            # 파일 이름에서 날짜와 번호 추출
            match = re.match(r'(\d{6})_(\d+)', file_name)
            if match:
                base_name = match.group(0)  # "231014_001" 형식
                input_image_path = os.path.join(INPUT_DIR, file_name)
                detect_and_segment_crops(input_image_path, OUTPUT_DIR, base_name)
            else:
                print(f"파일 이름 형식이 올바르지 않음: {file_name}")

    print("모든 작업 완료!")
