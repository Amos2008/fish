import os
import glob
import concurrent.futures
import time
import preImg
import cv2

def process_video(video_file):
    # 执行视频处理操作
    # ...
    start_time1 = time.time()
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_file}")
        # continue

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

    print(f"Processed video: {video_file}")


def process_videos_in_folder(folder_path):
    # 获取文件夹中所有视频文件的路径
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))  # 根据实际文件类型进行匹配

    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 提交视频处理任务给线程池
        # 每个视频文件都会分配一个线程进行处理
        for video_file in video_files:
            executor.submit(process_video, video_file)
start_time1 = time.time()
folder_path = "video/out"  # 替换为实际文件夹路径
process_videos_in_folder(folder_path)
end_time1 = time.time()
execution_time1 = end_time1 - start_time1
print(f"程序运行时间：{execution_time1}秒")
