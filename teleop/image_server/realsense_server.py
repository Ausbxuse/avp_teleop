import pyrealsense2 as rs
import time
import numpy as np
import cv2
import zmq
import zlib
import pickle
import struct
import datetime

# Configure depth and color streams
def start_server():
    # camera init
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    # Start streaming
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    print(vars(depth_intrinsics))

    print(f"Width: {depth_intrinsics.width}")
    print(f"Height: {depth_intrinsics.height}")
    print(f"Focal Length (fx, fy): ({depth_intrinsics.fx}, {depth_intrinsics.fy})")
    print(f"Principal Point (cx, cy): ({depth_intrinsics.cx}, {depth_intrinsics.cy})")
    print(f"Distortion Coefficients: {depth_intrinsics.coeffs}")

    # Set up ZeroMQ for sending data
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind("tcp://192.168.123.162:5556")  # Bind to all interfaces on port 5555
    print("The server has started, waiting for client connections...")

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = np.stack([depth_image] * 3, axis=-1)
        print("depth_iamge shape: ",depth_image.shape)
        color_image = np.asanyarray(color_frame.get_data())
        print("color_iamge shape: ",color_image.shape)

        # Optionally, process the depth image into a colormap
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Combine depth and color images for transmission (optional)
        combined_image = np.hstack((color_image, depth_image))
        # Compress the image for transmission
        _, encoded_image = cv2.imencode('.jpg', combined_image)  # Encode as JPEG
        compressed_data = zlib.compress(pickle.dumps(encoded_image))  # Compress

        chunk_size = 120000
        num_chunks = len(compressed_data) // chunk_size + 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = compressed_data[start:end]
            socket.send(chunk)
        print(datetime.datetime.now())

        ## Add frame header
        #frame_size = struct.pack('I', len(compressed_data))
        #print(len(frame_size))

        # Send the data
        #print(len(frame_size), len(compressed_data))
        #socket.send(frame_size)
        #socket.send(compressed_data)

    context.term()

if __name__ == "__main__":
    start_server()
