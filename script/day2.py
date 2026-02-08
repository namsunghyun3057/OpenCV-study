import cv2
import numpy as np

# 4. 텍스트
"""
img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (255, 255, 255)
THICKNESS = 1  # 선 두께
SCALE = 1  # 글자 크기

cv2.putText(
    img,
    "Hello OpenCV",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    SCALE,
    COLOR,
    THICKNESS,
)
# 위치, 내용, 시작 위치, 폰트, 크기, 색상, 두께
cv2.putText(
    img,
    "Nado Plain",
    (20, 150),
    cv2.FONT_HERSHEY_PLAIN,
    SCALE,
    COLOR,
    THICKNESS,
)
cv2.putText(
    img,
    "Nado Script Simplex",
    (20, 250),
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    SCALE,
    COLOR,
    THICKNESS,
)
cv2.putText(
    img,
    "Nado Triplex",
    (20, 350),
    cv2.FONT_HERSHEY_TRIPLEX,
    SCALE,
    COLOR,
    THICKNESS,
)
cv2.putText(
    img,
    "Nado Italic",
    (20, 450),
    cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC,
    SCALE,
    COLOR,
    THICKNESS,
)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 한글 우회 방법
"""

from PIL import ImageFont, ImageDraw, Image


def myPutText(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", font_size)
    draw.text(pos, text, font=font, fill=font_color)
    return np.array(img_pil)


img = np.zeros((480, 640, 3), dtype=np.uint8)

FONT_SIZE = 30
COLOR = (255, 255, 255)  # 흰색

img = myPutText(img, "안녕하세요. 반갑습니다.", (20, 50), FONT_SIZE, COLOR)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 5. 파일 저장

## 이미지 저장
"""
img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# result = cv2.imwrite("img_save.jpg", img)
result = cv2.imwrite("img_save.png", img)
print(result)
"""

## 비디오 저장
"""
cap = cv2.VideoCapture("video.mp4")

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*"DIVX")

width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))
# 파일명, 코덱, fps, 크기

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("더 이상 프레임이 없습니다.")
        break

    out.write(frame)  # 프레임 저장(소리 x)
    cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("q"):
        print("사용자에 의해 종료되었습니다.")
        break

out.release()
cap.release()
cv2.destroyAllWindows()
"""

# 6. 크기 조정

## 이미지
"""
img = cv2.imread("img.jpg")

### 고정 크기
dst = cv2.resize(img, (400, 500))  # 가로, 세로

### 비율로 조정
# 보간법
# 1. cv2.INTER_NEAREST : 가장 가까운 이웃 보간법
# 2. cv2.INTER_LINEAR : 양선형 보간법(기본값)
# 3. cv2.INTER_CUBIC : 4x4 픽셀 이웃을 사용한 3차 회선 보간법
# 4. cv2.INTER_LANCZOS4 : 8x8 픽셀 이웃을 사용한 Lanczos 보간법
# 5. cv2.INTER_AREA : 픽셀 영역 관계를 이용한 재샘플링 방법(축소에 좋음)
# 6. cv2.INTER_LINEAR_EXACT : 정확한 양선형 보간법
# 7. cv2.INTER_CUBIC_EXACT : 정확한 3차 회선 보간법
# 8. cv2.INTER_LANCZOS4_EXACT : 정확한 Lanczos 보간법
# 9. cv2.INTER_MAX : 최대 보간법
# 10. cv2.INTER_BICUBIC : BiCubic 보간법
dst = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

cv2.imshow("img", img)
cv2.imshow("resize", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

## 비디오
""""""
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("더 이상 프레임이 없습니다.")
        break

    # 고정 크기
    frame_resized = cv2.resize(frame, (400, 500))

    # 비율로 조정
    frame_resized = cv2.resize(
        frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )

    cv2.imshow("video", frame_resized)
    if cv2.waitKey(25) == ord("q"):
        print("사용자에 의해 종료되었습니다.")
        break

cap.release()
cv2.destroyAllWindows()
