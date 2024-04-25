import json
import struct
import time
import cv2
import numpy as np
import torch
import socket
import threading

class Fpi:
    def __init__(self, rtsp_cap, model_cfg, status):
        self.rtsp_cap = rtsp_cap
        self.stopped = False
        self.model = model_cfg
        self.status = status
        self.results_list = []

    def peo_detect(self):
        # while not self.stopped:
        #     if self.rtsp_cap.frame is not None:
        #         frame = self.rtsp_cap.frame
        #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        #         results = self.model(frame_rgb)
        #         # results.show()
        #         result_json = results.pandas().xyxy[0].to_json(orient="records")
        #         format_result_json = eval(result_json)
        #         print(format_result_json[0]['name'])
        #         print(format_result_json[0]['class'])
        #         self.results_list.append(format_result_json[0]['name'])
        #         # print(self.results_list)
        #         if len(self.results_list) >=2:
        #                 classlist = []
        #                 segment_size = 20
        #                 segments = [self.results_list[i:i + segment_size] for i in
        #                             range(0, len(self.results_list), segment_size)]
        #                 result = []
        #
        #                 for segment in segments:
        #                     counts = {}
        #                     for element in segment:
        #                         if element in counts:
        #                             counts[element] += 1
        #                         else:
        #                             counts[element] = 1
        #
        #                     most_common = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:1]
        #                     result.append(most_common)
        #
        #                 nc = int(len(result) / segment_size)
        #                 for i in range(nc):
        #                     if result[i][0][1] > 15:
        #                         element = result[i][0][0]
        #                         if element not in classlist:
        #                             classlist.append(element)
        #                 # --------------------------------------------------------------------------
        #
        #                 selected_classes = [i for i in classlist]
        #                 # selected_classes = [class_id for class_id, count in class_counts.items() if count >= threshold]
        #                 # print("类别出现次数统计：", class_counts)
        #                 # print("出现次数达到三次以上的类别：", selected_classes)
        #                 status1 = self.update_status(self.status, selected_classes)
        #                 print("更新后的status数组：", status1)
        #                 # print(bytearray(list(status1)))
        #
        #                 send_message_to_client(bytearray(list(status1)))
        #
        #                 # 清除降噪数据
        #                 self.results_list = []

            #-------------------------------------------------------------
            print('开始图像识别')
            while not self.stopped:
                if self.rtsp_cap.frame is not None:
                    frame = self.rtsp_cap.frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = self.model(frame_rgb)

                    result_json = results.pandas().xyxy[0].to_json(orient="records")
                    format_result_json = eval(result_json)

                    # print(format_result_json)
                    # print(format_result_json)

                    # ----------------识别加通信-----------------------------
                    # type_list = [0,0,0,0,0,0,0,0,0]     #图像识别（消防 + 穿戴 + 袖标 + 易燃物  +安全帽 防护面罩 手套  阴燃+人数）
                    class_list = []
                    count = 0
                    # 定义一个字典用于存储目标出现的次数
                    target_counts = {}

                    if format_result_json:
                        for i in range(len(format_result_json)):
                            # for (xmin, ymin, xmax, ymax) in format_result_json[i]:
                            #     cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 绘制绿色矩形框
                            target_class = format_result_json[i]['class']
                            if target_class not in target_counts:
                                target_counts[target_class] = [0, 0, 0]  # 初始化目标出现历史记录

                            # 更新目标出现历史记录
                            for j in range(3):
                                target_counts[target_class].append(1 if format_result_json[i]['class'] else 0)
                                # print(target_counts)
                                target_counts[target_class] = target_counts[target_class][-3:]  # 保留最近6帧记录
                            # print(target_counts[target_class])

                            for target_class, history in target_counts.items():
                                # print(history)
                                if sum(history) == 3:
                                    class_list.append(target_class)

                        print('检测到的目标class：',class_list)
                        check_groups = [
                            # [4, 5],  # 第一组
                            [4],
                            [6, 7, 8, 9],  # 第二组
                            [0],  # 第三组
                            [1, 2]  # 第四组
                        ]
                        type_list = []
                        for group in check_groups:
                            if all(item in class_list for item in group):
                                type_list.append(1)
                            else:
                                type_list.append(0)
                        type_list = type_list + [0, 0, 0, 0, 1]
                        type_list[2] = 1
                        type_list[1] = 1

                        for j in class_list:
                            if j == int(8):
                                type_list[4] = 1
                            if j == int(4) or j == int(5):
                                type_list[0] = 1
                            elif j == int(6):
                                type_list[5] = 1
                            elif j == int(7):
                                type_list[6] = 1

                            elif j == int(3):
                                count += 1
                        if count > 3:
                            type_list[8] = 0

                    else:
                        type_list = [0, 0, 0, 0, 0, 0, 0, 0, 1]
                    print(type_list)
                    # cv2.imshow(frame_rgb)
                    send_message_to_client(bytearray(type_list))
            #-------------------------------------------------------------

    def stop(self):
        self.stopped = True


class RTSPVideoCapture:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.frame = None
        self.stopped = False
        self.FPS = 1 / 30
        self.FPS_MS = int(self.FPS * 1000)

    def read(self):
        while not self.stopped:
            ret, self.frame = self.cap.read()
            # print(ret, self.frame)
            time.sleep(self.FPS)

    def stop(self):
        self.stopped = True


class MySocket:
    def __init__(self, socket):
        self.socket = socket
        self.is_connected = True

    def close(self):
        self.socket.close()
        self.is_connected = False

    def send(self, data):
        if self.is_connected:
            self.socket.send(data)
        else:
            raise Exception("Cannot send data, socket is closed.")


client_socket = None


# 用于处理客户端连接的函数
def handle_client(_client_socket):
    global client_socket
    while True:
        try:
            request = _client_socket.socket.recv(1024)
            if not request:
                print("Client closed connection.")
                _client_socket.close()
                client_socket = None
                break
            print(f"Received: {request}")
        except ConnectionResetError:
            print("Server closed connection.")
            _client_socket.close()
            client_socket = None
            break


# 你的服务器启动函数
def start_server():
    while True:
        time.sleep(1)
        server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(('10.5.8.69', 30050))
            server_socket.listen(1)  # Only expect one client
            print("Listening on port 1011")

            while True:
                try:
                    client, addr = server_socket.accept()
                    # client.settimeout(5)  # Set timeout for client socket
                    print(f"Accepted connection from: {addr[0]}:{addr[1]}")
                    global client_socket
                    client_socket = MySocket(client)
                    client_handler = threading.Thread(target=handle_client, args=(client_socket,))
                    client_handler.start()
                except Exception as e:
                    print(f"Error accepting client: {e}")
                    if server_socket:
                        server_socket.close()
                    break

        except Exception as e:
            print(f"Error: {e}. Restarting server.")
            if server_socket:
                server_socket.close()


# # 向客户端发送消息的函数
def send_message_to_client(body: bytes):
    if client_socket and isinstance(client_socket, MySocket):
        # 生成包头，它是一个 4 字节的整数，表示包体的长度
        # 小端：<I 大端：!I
        header = struct.pack("<I", len(body))
        # 将包头和包体组合成一个完整的消息
        message = header + body
        try:
            print(message)
            client_socket.send(message)
        except Exception as e:
            print(e)



if __name__ == '__main__':
    model_path = './best.pt'
    v5_path = 'E:/yolov5-slimming'
    choose_device = 'cuda:0'
    model = torch.hub.load(v5_path, 'custom', path=model_path, source='local', device=choose_device)
    model.conf = 0.4

    # 在另一个线程中启动服务器
    server_thread = threading.Thread(target=start_server)
    server_thread.start()
    # WIN_20231212_10_45_38_Pro   WIN_20231214_16_06_16_Pro
    video_path = r'E:\yolov5-slimming\testfiles\video\3.mp4'
    # video_path = r'E:\yolov5-slimming\testfiles\data_2570.jpg'
    rtsp_cap = RTSPVideoCapture(video_path)
    rtsp_cap_thread = threading.Thread(target=rtsp_cap.read)
    rtsp_cap_thread.start()

    status = np.full(shape=8, fill_value=99)
    status[0] = 200
    frame_processor = Fpi(rtsp_cap, model, status)
    rtsp_thread = threading.Thread(target=frame_processor.peo_detect)
    rtsp_thread.start()

