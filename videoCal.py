import preImg
import os
image_path = r"12.jpg"

import cv2
import glob

import time
# 在这里编写您的程序代码
# 设置视频文件所在的目录路径
video_dir = "video/one"

# 获取目录中所有的视频文件路径
video_files = glob.glob(video_dir + "/*.mp4")  # 根据实际视频文件的扩展名进行调整

# 遍历视频文件列表
for video_file in video_files:

    start_time1 = time.time()
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_file}")
        continue

    # 获取视频文件信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    print(f"处理视频文件: {video_file}")
    print(f"帧数: {frame_count}")
    print(f"帧率: {fps}")
    print(f"时长: {duration} 秒")

    nums = []
    count = 0
    # 遍历视频的每一帧
    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 检查是否成功读取帧
        if not ret:
            break
        # 只处理每隔5帧的帧
        if count % 50 == 0:
            start_time = time.time()
            num = 0
            preImg.preImg1(frame, num)
            nums.append(num)

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"程序运行时间：{execution_time}秒")
        count += 1
        # 在这里处理每一帧，例如进行图像处理、分析等
    print(count)
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    print(f"程序运行时间：{execution_time1}秒")

    # 释放视频对象和关闭窗口
    cap.release()
    cv2.destroyAllWindows()

