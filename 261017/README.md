# 2026년 10월 17일 Daily Assignment02

모폴로지 연산과 외곽선 검출을 활용한 농구공 영역 추출 과제입니다.

## 의존성

- Python 3
- NumPy
- OpenCV (`cv2`)
- Matplotlib

PC에서 필요한 패키지 설치:

```sh
python -m pip install numpy opencv-python matplotlib
```

Jetson에서는 JetPack에 포함된 OpenCV를 사용하고, 없는 패키지만 추가 설치합니다.

## 실행 방법

워크스페이스 최상위 폴더에서 다음 명령을 실행합니다.

```sh
cd 261017
python Daily_Assignment02.py
```

## 참고 사항

- `261017` 폴더 안에서 실행해야 합니다.
- 입력 이미지는 상위 폴더의 `src/images/basketball_crop.jpg`이며, 코드에서는 `../src/images/basketball_crop.jpg`로 참조합니다.
- 별도 사용자 입력 없이 실행되며, 처리 단계와 검출 결과가 Matplotlib 창에 표시됩니다.
