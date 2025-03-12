import datetime
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# Shared variables for the latest frame
latest_frame_bytes = None
frame_lock = threading.Lock()


def frame_capture_thread(pipeline):
    global latest_frame_bytes
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)

        if not (depth_frame and color_frame and ir_left_frame and ir_right_frame):
            continue

        # Convert frames to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())

        # Convert single-channel images to 3-channel images for consistency
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        ir_left_image = cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR)
        ir_right_image = cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)

        # Combine the images horizontally
        combined_image = np.hstack(
            (color_image, depth_image, ir_left_image, ir_right_image)
        )

        # JPEG encode the combined image (JPEG compression is fast)
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        # ret, encoded_image = cv2.imencode(".jpg", combined_image, encode_param)
        ret, encoded_image = cv2.imencode(".jpg", combined_image)
        if ret:
            frame_bytes = encoded_image.tobytes()
            with frame_lock:
                latest_frame_bytes = frame_bytes


def start_server():
    # Initialize RealSense pipeline and configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)

    # Start the background frame capture thread
    capture_thread = threading.Thread(
        target=frame_capture_thread, args=(pipeline,), daemon=True
    )
    capture_thread.start()

    # Initialize ZeroMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://192.168.123.162:5556")

    print("Server started, waiting for client requests...")
    try:
        while True:
            request = socket.recv()
            print(f"Received request: {request.decode()} at {datetime.datetime.now()}")

            with frame_lock:
                frame_bytes = latest_frame_bytes

            if frame_bytes is None:
                print("No frame available yet.")
                socket.send(b"")  # or send an error indicator
            else:
                socket.send(frame_bytes)
                print(f"Sent frame at {datetime.datetime.now()}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_server()
