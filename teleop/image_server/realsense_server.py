import datetime
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# Shared variables for the latest frames
latest_rgb_ir_bytes = None
latest_depth_bytes = None
frame_lock = threading.Lock()


def frame_capture_thread():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)

    global latest_rgb_ir_bytes, latest_depth_bytes

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
        depth_array = np.asanyarray(depth_frame.get_data()).astype(
            np.uint16
        )  # Keep as float32
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())

        # Convert single-channel IR images to 3-channel images for consistency
        ir_left_image = cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR)
        ir_right_image = cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)

        # Combine the RGB and IR images horizontally
        rgb_ir_combined = np.hstack((color_image, ir_left_image, ir_right_image))

        # JPEG encode the RGB+IR combined image
        ret, encoded_rgb_ir = cv2.imencode(".jpg", rgb_ir_combined)

        if ret:
            # Prepare both parts
            rgb_ir_bytes = encoded_rgb_ir.tobytes()
            depth_bytes = depth_array.tobytes()  # Raw bytes of float32 depth data

            with frame_lock:
                latest_rgb_ir_bytes = rgb_ir_bytes
                latest_depth_bytes = depth_bytes


def start_server():
    # Start the background frame capture thread
    capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
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
                rgb_ir_bytes = latest_rgb_ir_bytes
                depth_bytes = latest_depth_bytes

            if rgb_ir_bytes is None or depth_bytes is None:
                print("No frames available yet.")
                socket.send_multipart([b"", b""])  # Send empty parts
            else:
                # Send as multipart message: [rgb_ir_jpeg, raw_depth_float32]
                socket.send_multipart([rgb_ir_bytes, depth_bytes])
                print(f"Sent multipart frame at {datetime.datetime.now()}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_server()
