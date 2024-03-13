import cv2

# 打开原始视频
video_path = 'video/subvideo.mp4'
video_capture = cv2.VideoCapture(video_path)

# 获取视频的基本信息
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计算每个子视频的帧数
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frames_per_subvideo = total_frames // 10

# 逐个拆分子视频
for i in range(10):
    # 创建子视频文件名
    subvideo_path = 'video/out/subvideo_{}.mp4'.format(i + 1)

    # 创建子视频编写器
    subvideo_writer = cv2.VideoWriter(subvideo_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 写入子视频的帧
    start_frame = i * frames_per_subvideo
    end_frame = start_frame + frames_per_subvideo

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while start_frame < end_frame and video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            subvideo_writer.write(frame)
            start_frame += 1
        else:
            break

    # 释放子视频编写器
    subvideo_writer.release()

# 释放原始视频捕获器
video_capture.release()