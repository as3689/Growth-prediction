import cv2
import numpy as np
import os

def refine_crop(crop):
    """
    이미지를 정제하여 녹색 작물만 남기고, 테두리에 찍힌 작물은 제거합니다.
    """
    # 이미지 크기
    h, w = crop.shape[:2]

    # HSV 색공간으로 변환 (녹색 추출용)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 녹색 범위 설정
    lower_green = np.array([30, 30, 30])  # 약간 더 넓은 범위로 설정
    upper_green = np.array([90, 255, 255])

    # 녹색 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 테두리 영역에 있는 녹색 픽셀 제거
    border_thickness = 5  # 테두리 두께
    mask[:border_thickness, :] = 0  # 상단 테두리 제거
    mask[-border_thickness:, :] = 0  # 하단 테두리 제거
    mask[:, :border_thickness] = 0  # 좌측 테두리 제거
    mask[:, -border_thickness:] = 0  # 우측 테두리 제거

    # 마스크 확장 및 구멍 채우기
    kernel_close = np.ones((9, 9), np.uint8)  # 구멍 채우기용 커널 (이전보다 큼)
    kernel_dilate = np.ones((11, 11), np.uint8)  # 확장용 커널 (이전보다 큼)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)  # 구멍 채우기
    mask = cv2.dilate(mask, kernel_dilate, iterations=3)  # 마스크 확장으로 중앙 빈 공간 채우기 (반복 증가)

    # 마스크를 부드럽게 처리
    mask = cv2.GaussianBlur(mask, (7, 7), 0)  # 블러 강도 약간 증가

    # 마스크가 정상적으로 작물 영역을 잡았는지 확인
    if np.sum(mask) < 1000:  # 마스크 픽셀 합이 너무 작으면
        raise ValueError("마스크 감지 실패: 녹색 범위를 확인하세요!")

    # 원본 색상을 유지하면서 마스크 적용
    result = cv2.bitwise_and(crop, crop, mask=mask)

    # 알파 채널 추가 (투명 배경)
    b, g, r = cv2.split(result)
    alpha = mask  # 마스크를 알파 채널로 사용
    result_with_alpha = cv2.merge((b, g, r, alpha))

    return result_with_alpha

def process_images(input_dir, output_dir, start_img_counter=1):
    """
    입력 디렉토리의 이미지를 처리하여 정제된 이미지를 출력 디렉토리에 저장.
    """
    os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성
    img_counter = start_img_counter  # 시작 번호를 지정

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.png')):  # 지원하는 이미지 파일 확장자
            input_image_path = os.path.join(input_dir, file_name)

            # 이미지 로드
            crop = cv2.imread(input_image_path)

            try:
                # 이미지 정제
                refined_crop = refine_crop(crop)

                # 파일명 생성 (img001, img002, ...)
                output_file_name = f"img{img_counter:03d}.png"
                output_path = os.path.join(output_dir, output_file_name)

                # 결과 저장
                cv2.imwrite(output_path, refined_crop)
                print(f"{output_file_name} 저장 완료")

                img_counter += 1  # 이미지 번호 증가
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("모든 작업 완료!")

# 실행 코드
if __name__ == '__main__':
    INPUT_DIR = 'data/images/10월24일'  # 원본 이미지가 저장된 디렉토리
    OUTPUT_DIR = 'data/images/images'  # 정제된 이미지를 저장할 디렉토리
    START_NUMBER = 165  # 저장 시작 번호 설정 (예: 10 → img010부터 시작)

    process_images(INPUT_DIR, OUTPUT_DIR, start_img_counter=START_NUMBER)
