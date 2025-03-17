import datetime
import os
import pickle
import threading
import time
import zlib

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# Shared variables for the latest frames
latest_frame_bytes = None  # for the combined JPEG image (color + IR)
latest_depth_bytes = None  # for the depth image as pickled compressed npy
frame_lock = threading.Lock()


def frame_capture_thread():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)

    global latest_frame_bytes, latest_depth_bytes
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

        # For display, convert IR images from single-channel to BGR.
        ir_left_bgr = cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR)
        ir_right_bgr = cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)

        # Combine color and IR images horizontally (depth is sent separately)
        combined_img = np.hstack((color_image, ir_left_bgr, ir_right_bgr))
        ret, encoded_image = cv2.imencode(".jpg", combined_img)
        if ret:
            jpg_frame_bytes = encoded_image.tobytes()
        else:
            jpg_frame_bytes = b""

        # For depth, we send the raw depth image as a pickled and compressed npy.
        compressed_depth = zlib.compress(pickle.dumps(depth_image))

        with frame_lock:
            latest_frame_bytes = jpg_frame_bytes
            latest_depth_bytes = compressed_depth


def start_server():
    # Start the background frame capture thread.
    capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
    capture_thread.start()

    # Initialize ZeroMQ REP (reply) socket.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # Replace the IP with your server's IP as needed.
    socket.bind("tcp://192.168.123.162:5556")

    print("Server started, waiting for client requests...")
    try:
        while True:
            request = socket.recv()
            print(f"Received request: {request.decode()} at {datetime.datetime.now()}")

            with frame_lock:
                jpg_frame = latest_frame_bytes
                depth_frame = latest_depth_bytes

            if jpg_frame is None or depth_frame is None:
                print("No frame available yet.")
                socket.send_multipart([b""])
            else:
                # Send as a multipart message: first part is the JPEG-encoded combined image,
                # second part is the pickled & compressed depth image.
                socket.send_multipart([jpg_frame, depth_frame])
                print(f"Sent frames at {datetime.datetime.now()}")
            time.sleep(0.01)  # slight delay to avoid busy-waiting
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_server()
