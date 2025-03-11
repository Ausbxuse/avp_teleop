import datetime
import pickle
import zlib

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq


def start_server():
    # 初始化 RealSense 设备
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用 RGB、深度 和 左右红外摄像头
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色图像
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度图像
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # 左 IR
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # 右 IR

    # 启动流
    pipeline.start(config)

    # ZeroMQ 服务器初始化
    context = zmq.Context()
    socket = context.socket(zmq.REP)

    # socket.setsockopt(zmq.SNDHWM, 1)

    socket.bind("tcp://192.168.123.162:5556")  # 绑定端口

    print("The server has started, waiting for client connections...")

    try:
        while True:

            request = socket.recv()
            print(f"Received request: {request.decode()} at {datetime.datetime.now()}")

            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_left_frame = frames.get_infrared_frame(1)  # 左 IR
            ir_right_frame = frames.get_infrared_frame(2)  # 右 IR

            if (
                not depth_frame
                or not color_frame
                or not ir_left_frame
                or not ir_right_frame
            ):
                print("got no frame")
                continue

            # 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())  # (480, 640, 3)
            depth_image = np.asanyarray(depth_frame.get_data())  # (480, 640)
            ir_left_image = np.asanyarray(ir_left_frame.get_data())  # (480, 640)
            ir_right_image = np.asanyarray(ir_right_frame.get_data())  # (480, 640)

            # 把深度图、左右 IR 图像转换为 3 通道 (H, W) → (H, W, 3)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            ir_left_image = cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR)
            ir_right_image = cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)

            # 合并所有图像用于传输
            combined_image = np.hstack(
                (color_image, depth_image, ir_left_image, ir_right_image)
            )

            # 压缩并序列化数据
            _, encoded_image = cv2.imencode(".jpg", combined_image)  # JPEG 编码
            compressed_data = zlib.compress(pickle.dumps(encoded_image))  # 压缩

            # 发送数据
            chunk_size = 240000

            # num_chunks = len(compressed_data) // chunk_size + 1
            # for i in range(num_chunks):
            #     start = i * chunk_size
            #     end = start + chunk_size
            #     chunk = compressed_data[start:end]
            #     socket.send(chunk)

            chunks = [
                compressed_data[i : i + chunk_size]
                for i in range(0, len(compressed_data), chunk_size)
            ]
            socket.send_multipart(chunks)

            print(f"Sent frame at {datetime.datetime.now()}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_server()
