import zmq
import zlib
import pickle
import cv2
import numpy as np

def start_receiver():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")  

    print("Receiver is running, waiting for frames...")

    compressed_data = b""
    chunk_size = 120000

    while True:
        try:
            compressed_data = b''
            while True:
                chunk = socket.recv()
                compressed_data += chunk
                if len(chunk) < 60000:
                    break

            data = zlib.decompress(compressed_data)
            frame_data = pickle.loads(data)

            # 解码 JPEG
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

            height, width, _ = frame.shape
            width_each = width // 4

            color_image = frame[:, 0:width_each]
            depth_image = frame[:, width_each:2*width_each]
            ir_left_image = frame[:, 2*width_each:3*width_each]
            ir_right_image = frame[:, 3*width_each:]

            # 显示
            cv2.imshow("RGB Image", color_image)
            cv2.imshow("Depth Image", depth_image)
            cv2.imshow("Left IR", ir_left_image)
            cv2.imshow("Right IR", ir_right_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("Error receiving frame:", e)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_receiver()
