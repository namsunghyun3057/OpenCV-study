import cv2
import numpy as np

# 7. 이미지 자르기

## 영역을 잘라서 새로운 창에 표시
"""
img = cv2.imread("img.jpg")
# print(img.shape)  # (391, 640, 3) 세로, 가로, 채널

crop = img[100:200, 200:400]  # 세로 기준 100 : 200, 가로 기준 300 : 400 까지 자름

cv2.imshow("img", img)
cv2.imshow("crop", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 영역을 잘라서 기존 창에 표시
"""
img = cv2.imread("img.jpg")

crop = img[100:200, 200:400]  # 세로 기준 100 : 200, 가로 기준 300 : 400 까지 자름
img[100:200, 400:600] = (
    crop  # 잘라낸 영역을 다시 원래 이미지의 해당 위치(세로, 가로)에 넣음
)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 8. 이미지 대칭

## 좌우 대칭
"""
img = cv2.imread("img.jpg")
flip_horizontal = cv2.flip(img, 1)  # flipCode > 0 : 좌우 대칭(horizontal)

cv2.imshow("img", img)
cv2.imshow("flip_horizontal", flip_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 상하 대칭
"""
img = cv2.imread("img.jpg")
flip_vertical = cv2.flip(img, 0)  # flipCode = 0 : 상하 대칭(vertical)

cv2.imshow("img", img)
cv2.imshow("flip_vertical", flip_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 상하좌우 대칭
"""
img = cv2.imread("img.jpg")
flip_both = cv2.flip(img, -1)  # flipCode < 0 : 상하좌우 대칭(both)
cv2.imshow("img", img)
cv2.imshow("flip_both", flip_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 9. 이미지 회전

## 시계 방향 90도 회전
"""
img = cv2.imread("img.jpg")

rotate_90 = cv2.rotate(
    img, cv2.ROTATE_90_CLOCKWISE
)  # cv2.ROTATE_90_CLOCKWISE : 시계 방향 90도 회전

cv2.imshow("img", img)
cv2.imshow("rotate_90", rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 180도 회전
"""
img = cv2.imread("img.jpg")

rotate_180 = cv2.rotate(img, cv2.ROTATE_180)  # cv2.ROTATE_180 : 180도 회전

cv2.imshow("img", img)
cv2.imshow("rotate_180", rotate_180)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 반시계 방향 90도 회전
"""
img = cv2.imread("img.jpg")
rotate_90_counter = cv2.rotate(
    img, cv2.ROTATE_90_COUNTERCLOCKWISE
)  # cv2.ROTATE_90_COUNTERCLOCKWISE : 반시계 방향 90도 회전
cv2.imshow("img", img)
cv2.imshow("rotate_90_counter", rotate_90_counter)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 10. 이미지 변형 (흑백)

## 이미지를 흑백으로 불러오기 -> 1. 이미지 출력 참고
## 불러온 이미지를 흑백으로 변형
"""
img = cv2.imread("img.jpg")

dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv2.COLOR_BGR2GRAY : BGR -> GRAY(흑백)

cv2.imshow("img", img)
cv2.imshow("gray", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 11. 이미지 변형 (흐림)
# 가우시안 블러

## 커널 사이즈 변화에 따른 흐림
"""
img = cv2.imread("img.jpg")

kernel_3 = cv2.GaussianBlur(img, (3, 3), 0)  # 커널 크기 3x3
kernel_5 = cv2.GaussianBlur(img, (5, 5), 0)  # 커널 크기 5x5
kernel_7 = cv2.GaussianBlur(img, (7, 7), 0)  # 커널 크기 7x7

cv2.imshow("img", img)
cv2.imshow("kernel_3", kernel_3)
cv2.imshow("kernel_5", kernel_5)
cv2.imshow("kernel_7", kernel_7)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 표준 편차 변화에 따른 흐림
"""
img = cv2.imread("img.jpg")

sigma_1 = cv2.GaussianBlur(
    img, (0, 0), 1
)  # sigmaX - 가우시안 커널의 x 방향의 표준 편차
sigma_2 = cv2.GaussianBlur(img, (0, 0), 2)  # 커널 크기 5x5
sigma_3 = cv2.GaussianBlur(img, (0, 0), 3)  # 커널 크기 7x7

cv2.imshow("img", img)
cv2.imshow("sigma_1", sigma_1)
cv2.imshow("sigma_2", sigma_2)
cv2.imshow("sigma_3", sigma_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 12. 이미지 변형 (원근)

## 사다리꼴 이미지 세우기
"""
img = cv2.imread("newspaper.jpg")

width, height = 640, 300

src = np.array(
    [[511, 352], [1008, 345], [1122, 584], [455, 594]], dtype=np.float32
)  # input points
dst = np.array(
    [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
)  # output points
# 좌상, 우상, 우하, 좌하 (시계방향으로 정의)

matrix = cv2.getPerspectiveTransform(src, dst)  # matrix 계산
result = cv2.warpPerspective(img, matrix, (width, height))  # matrix 대로 원근 변환 적용

cv2.imshow("img", img)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
## 회전된 이미지 세우기
""""""
img = cv2.imread("poker.jpg")

width, height = 530, 710

src = np.array(
    [[702, 143], [1133, 414], [726, 1007], [276, 700]], dtype=np.float32
)  # input points
dst = np.array(
    [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
)  # output points
# 좌상, 우상, 좌하, 우하 (시계방향으로 정의)

matrix = cv2.getPerspectiveTransform(src, dst)  # matrix 계산
result = cv2.warpPerspective(img, matrix, (width, height))  # matrix 대로 원근 변환 적용

cv2.imshow("img", img)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
