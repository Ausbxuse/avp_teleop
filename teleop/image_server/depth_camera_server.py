import pickle
import zlib

import cv2
import pyrealsense2 as rs
import zmq

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()


def start_server():
    try:
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue

            # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
            coverage = [0] * 64
            for y in range(480):
                for x in range(640):
                    dist = depth.get_distance(x, y)
                    if 0 < dist and dist < 1:
                        coverage[x // 10] += 1

                if y % 20 is 19:
                    line = ""
                    for c in coverage:
                        line += " .:nhBXWW"[c // 25]
                    coverage = [0] * 64
                    print(line)

    finally:
        pipeline.stop()
    # camera init
    cap = cv2.VideoCapture(3, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # set ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind("tcp://*:5555")
    print("The server has started, waiting for client connections...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame read is error")
            break

        # encoding image
        ret2, frame = cv2.imencode(".jpg", frame)
        if not ret2:
            continue

        # Compressing data using pickle and zlib
        data = pickle.dumps(frame)
        compressed_data = zlib.compress(data)

        # sending data in pieces
        chunk_size = 60000
        num_chunks = len(compressed_data) // chunk_size + 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = compressed_data[start:end]
            socket.send(chunk)
    cap.release()
    context.term()


if __name__ == "__main__":
    start_server()
