import cv2
import time

def detect_keypoints(image, method="SIFT"):
    # 特徴点検出アルゴリズムの選択
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "AKAZE":
        detector = cv2.AKAZE_create()
    elif method == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Unknown method: Choose 'SIFT', 'AKAZE', or 'ORB'")
    
    # 特徴点の検出開始
    start_time = time.time()  # 計測開始
    keypoints, descriptors = detector.detectAndCompute(image, None)
    elapsed_time = time.time() - start_time  # 処理にかかった時間
    return keypoints, elapsed_time

# 画像の読み込み
image = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)

# アルゴリズムごとの特徴点検出と処理時間の計測
for method in ["SIFT", "AKAZE", "ORB"]:
    keypoints, elapsed_time = detect_keypoints(image, method=method)
    print(f"{method}: {len(keypoints)} keypoints detected in {elapsed_time:.4f} seconds")

    # 特徴点の可視化と保存
    output_image = cv2.drawKeypoints(image, keypoints, None)
    output_filename = f"output_{method}_{len(keypoints)}_keypoints.png"
    cv2.imwrite(output_filename, output_image)
    print(f"Saved output image to {output_filename}")
