import cv2
import numpy as np

# 1. 이미지 출력
"""
img = cv2.imread("img.jpg")
cv2.imshow("img", img)
key = cv2.waitKey(0)
print(key)
cv2.destroyAllWindows()

# 다양한 옵션으로 이미지 읽기
img_color = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread("img.jpg", cv2.IMREAD_UNCHANGED)

cv2.imshow("img_color", img_color)
cv2.imshow("img_gray", img_gray)
cv2.imshow("img_unchanged", img_unchanged)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지의 크기 확인
img = cv2.imread("img.jpg")
print(img.shape)
"""

# 2. 비디오 출력
"""
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("더 이상 프레임이 없습니다.")
        break
    cv2.imshow("video", frame)
    if cv2.waitKey(25) == ord("q"):
        print("사용자에 의해 종료되었습니다.")
        break

cap.release()
cv2.destroyAllWindows()
"""

# 3. 카메라 출력
"""
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) == ord("q"):
        print("사용자에 의해 종료되었습니다.")
        break
cap.release()
cv2.destroyAllWindows()
"""

# 4. 도형 그리기
"""
img = np.zeros((480, 640, 3), dtype=np.uint8)
# img[:] = (255, 255, 255) # openCV는 BGR 순서
# print(img)
# img[100:200, 200:300] = (255, 255, 255)
# [세로 영역, 가로 영역]

# 선 그리기
COLOR = (0, 255, 255)  # BGR
THICKNESS = 3
cv2.line(img, (50, 100), (400, 50), COLOR, THICKNESS, cv2.LINE_AA)

# 원 그리기
COLOR = (255, 255, 0)  # BGR
RADIUS = 50
THICKNESS = 10
cv2.circle(img, (300, 300), RADIUS, COLOR, THICKNESS, cv2.LINE_AA)
cv2.circle(img, (400, 300), RADIUS, COLOR, cv2.FILLED, cv2.LINE_AA)

# 사각형 그리기
COLOR = (255, 0, 255)  # BGR
THICKNESS = 3
cv2.rectangle(img, (100, 200), (300, 400), COLOR, THICKNESS, cv2.LINE_AA)
# 그릴 위치, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색상, 두께, 선 종류
cv2.rectangle(img, (120, 220), (280, 380), COLOR, cv2.FILLED, cv2.LINE_AA)

# 다각형 그리기
COLOR = (0, 255, 0)  # BGR
THICKNESS = 3
pts1 = np.array([[400, 200], [500, 250], [450, 350], [350, 300]])
pts2 = np.array([[500, 100], [600, 150], [550, 250]], np.int32)
# cv2.polylines(img, [pts1], isClosed=True, color=COLOR, thickness=THICKNESS, lineType=cv2.LINE_AA)
# cv2.polylines(img, [pts2], isClosed=True, color=COLOR, thickness=THICKNESS, lineType=cv2.LINE_AA)
cv2.polylines(
    img,
    [pts1, pts2],
    isClosed=True,
    color=COLOR,
    thickness=THICKNESS,
    lineType=cv2.LINE_AA,
)

pts3 = np.array(
    [[[100, 50], [200, 100], [150, 200]], [[100, 150], [300, 100], [100, 200]]],
    np.int32,
)
cv2.fillPoly(img, pts3, COLOR, cv2.LINE_AA)

star_pts = np.array(
    [
        [250, 100],  # 맨 위 (바깥)
        [285, 200],  # (안쪽)
        [390, 200],  # 오른쪽 (바깥)
        [305, 265],  # (안쪽)
        [340, 370],  # 오른쪽 아래 (바깥)
        [250, 310],  # (안쪽)
        [160, 370],  # 왼쪽 아래 (바깥)
        [195, 265],  # (안쪽)
        [110, 200],  # 왼쪽 (바깥)
        [215, 200],  # (안쪽)
    ],
    dtype=np.int32,
)
cv2.fillPoly(img, [star_pts], COLOR, cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
