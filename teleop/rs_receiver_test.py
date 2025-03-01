import pickle
import time
import zlib

import cv2
import zmq

frame_width = 1280  # Set expected width
frame_height = 480  # Set expected height
fps = 30  # Frames per second
output_filename = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
video_writer = cv2.VideoWriter(
    output_filename, fourcc, fps, (frame_width, frame_height)
)


def rs_receiver():
    # Set up ZeroMQ for receiving data
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")  # Replace with the robot's IP and port

    while True:
        compressed_data = b""
        while True:
            chunk = socket.recv()
            compressed_data += chunk
            if len(chunk) < 120000:
                break
        data = zlib.decompress(compressed_data)
        frame_data = pickle.loads(data)

        # Decode and display the image
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        print(frame.shape)
        # cv2.imshow("test", frame)
        video_writer.write(frame)

        # Optional: Show frame (for debugging)
        # cv2.imshow("Live Stream", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # sm_rs.write_image(frame)
        # Control receiving frequency
        time.sleep(0.02)


rs_receiver()

video_writer.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_filename}")


# Cleanup
