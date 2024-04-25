# -*- coding: GBK -*-
import ctypes
import time
from ctypes import create_string_buffer

import cv2
import numpy as np
import requests
import torch
import os
from numpy import empty
from collections import Counter
# from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from HCNetSDK import *
from PlayCtrl import *
import face_recognition
from miio.device import Device
import threading
import socket
import select
import json
import random
import serial
import math
from insightface.app import FaceAnalysis

w, h = 960, 540

class CameraSDK:

    def __init__(self, dev_ip, dev_port, dev_user_name, dev_password):
        self.DEV_IP = create_string_buffer(dev_ip.encode('utf-8'))
        self.DEV_PORT = dev_port
        self.DEV_USER_NAME = create_string_buffer(dev_user_name.encode('utf-8'))
        self.DEV_PASSWORD = create_string_buffer(dev_password.encode('utf-8'))
        self.lUserId = 0

        os.chdir(r'./lib/win')
        self.Objdll = ctypes.CDLL(r'./HCNetSDK.dll')  # ���������
        self.Playctrldll = ctypes.CDLL(r'./PlayCtrl.dll')  # ���ز��ſ�
        self.Objdll.NET_DVR_Init()

        # self.RealDataCallBack_V30 = CFUNCTYPE(None, c_long, c_uint, POINTER(c_byte), c_uint, c_void_p)
        self.RealDataCallBack_V30 = fun_ctype(None, c_long, c_ulong, POINTER(c_ubyte), c_ulong, c_void_p)
        self.deccbfunwin = fun_ctype(None, c_long, POINTER(c_char), c_long, POINTER(FRAME_INFO), c_void_p, c_void_p)

        self.PlayCtrl_Port = c_long(-1)

        self.image_data = None  # Initialize image_data
        self.preview_info = NET_DVR_PREVIEWINFO()

    def login_dev(self):
        device_info = NET_DVR_DEVICEINFO_V30()
        self.lUserId = self.Objdll.NET_DVR_Login_V30(self.DEV_IP, self.DEV_PORT, self.DEV_USER_NAME, self.DEV_PASSWORD,
                                                     byref(device_info))
        return (self.lUserId, device_info)

    def real_data_call_back_V30(self, lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
        # print("real_data_call_back_V30 called")
        if dwDataType == NET_DVR_SYSHEAD:
            # print("�����˵�һ��")
            self.Playctrldll.PlayM4_SetStreamOpenMode(self.PlayCtrl_Port, 0)
            if self.Playctrldll.PlayM4_OpenStream(self.PlayCtrl_Port, pBuffer, dwBufSize, 1024 * 1024):
                # print("�����˵ڶ���")
                if not self.Playctrldll.PlayM4_Play(self.PlayCtrl_Port):
                    print('���ſⲥ��ʧ��')
                else:
                    # Save the image data
                    # self.image_data = np.frombuffer(pBuffer, dtype=np.uint8)

                    global FuncDecCB
                    FuncDecCB = self.deccbfunwin(self.DecCBFun)
                    self.Playctrldll.PlayM4_SetDecCallBackExMend(self.PlayCtrl_Port, FuncDecCB, None, 0, None)
                    if self.Playctrldll.PlayM4_Play(self.PlayCtrl_Port):
                        # print("�����˵�����")
                        print(u'���ſⲥ�ųɹ�')
                    else:
                        print(u'���ſⲥ��ʧ��')
            else:
                print('���ſ����ʧ��')
        elif dwDataType == NET_DVR_STREAMDATA:
            # print("������elif")
            self.Playctrldll.PlayM4_InputData(self.PlayCtrl_Port, pBuffer, dwBufSize)
        else:
            print("������else")

    def DecCBFun(self, nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
        # print("�����ˣ��ص�����")
        # ����ص�����
        if pFrameInfo.contents.nType == 3:
            # ���뷵����ƵYUV���ݣ���YUV����ת��jpgͼƬ���浽����
            # ����к�ʱ������Ҫ���������ݿ������ص���������������߳����洦�����������ص����½��붪֡
            nWidth = pFrameInfo.contents.nWidth
            nHeight = pFrameInfo.contents.nHeight

            nparr = np.frombuffer(pBuf[:nSize], dtype=np.uint8)
            img_dst = np.reshape(nparr, [nHeight + nHeight // 2, nWidth])
            img_rgb = cv2.cvtColor(img_dst, cv2.COLOR_YUV2BGR_YV12)
            self.image_data = img_rgb
            img_rgb_rs = cv2.resize(img_rgb, (960, 540))
            cv2.imshow("camera", img_rgb_rs)
            cv2.waitKey(1)

    def display_image(self):
        while True:
            if self.image_data is not None:
                cv2.imshow("Camera Image", self.image_data)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                print('ceshi')

    def start_preview(self):
        # ��ȡһ�����ž��
        if not self.Playctrldll.PlayM4_GetPort(byref(self.PlayCtrl_Port)):
            print(u'��ȡ���ſ���ʧ��')
        # print('run')
        RealDataCallBack_V30 = self.RealDataCallBack_V30(self.real_data_call_back_V30)
        self.preview_info.hPlayWnd = 0
        self.preview_info.lChannel = 1  # ͨ����
        self.preview_info.dwStreamType = 0  # ������
        self.preview_info.dwLinkMode = 0  # TCP
        self.preview_info.bBlocked = 1  # ����ȡ��
        lPreviewHandle = self.Objdll.NET_DVR_RealPlay_V40(self.lUserId, byref(self.preview_info), RealDataCallBack_V30,
                                                          None)
        while True:
            lRet = self.Objdll.NET_DVR_PTZControl(lPreviewHandle, PAN_LEFT, 0)
        # lRet = self.Objdll.NET_DVR_PTZControl(lPreviewHandle, PAN_LEFT, 0)

        return lPreviewHandle

    def run(self):
        self.login_dev()
        self.start_preview()

        while True:
            time.sleep(1)

    def start_thread(self):
        threading.Thread(target=self.run).start()

lock = threading.Lock()

class FrameProcessor:

    def __init__(self, camera_200, model_cfg, verification_completed_event, shared_state):
        self.camera_200 = camera_200
        # self.camera_201 = camera_201
        self.stopped = False
        self.model = model_cfg
        self.w, self.h = 960, 540
        self.shared_state = shared_state
        self.verification_completed_event = verification_completed_event

    def process(self):
        # self.verification_completed_event.wait()
        while not self.stopped:
            if self.camera_200.image_data is not None:
                frame = self.camera_200.image_data
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                T2 = time.time()

                print('��ʼͼ��ʶ��')

                results_yolo = self.model(source=frame)
                results = results_yolo[0]
                cls = results.boxes.cls
                cls_int = cls.int()
                classes_list_int = cls_int.tolist()

                #45, 6789, 0, 12
                # classes_list_int = cls_int.tolist()
                check_groups = [
                    [4, 5],  # ��һ��
                    [6, 7, 8, 9],  # �ڶ���
                    [0],  # ������
                    [1, 2]  # ������
                ]
                type_list = []
                for group in check_groups:

                    if all(item in classes_list_int for item in group):
                        type_list.append(1)
                    else:
                        type_list.append(0)
                print(type_list)
                # lock.acquire()
                # try:
                #     if type_list[0] == 1 and type_list[2] == 1 and type_list[3] == 0:
                #         plug.send("set_properties", [{'did': 'MYDID', 'siid': 2, 'piid': 1, 'value': True}])
                #         self.shared_state['plug_state'] = True
                #     else:
                #         plug.send("set_properties", [{'did': 'MYDID', 'siid': 2, 'piid': 1, 'value': False}])
                #         self.shared_state['plug_state'] = False
                #     print('1:',self.shared_state['plug_state'])
                # finally:
                #     lock.release()
                # cv2.imshow()
                # if self.shared_state['plug_state'] == True:
                send_message_to_server_1(type_list)
                    # send_message_to_server_qiti(self.verification_completed_event, self.shared_state)

                T3 = time.time()
                print('������������ʱ��:%s����' % ((T3 - T2) * 1000))
                # self.verification_completed_event.set()


                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop()

    def stop(self):
        self.stopped = True

class MySocket:
    def __init__(self, socket):
        self.socket = socket

    def send(self, data, index):

        print(f'{index}���͵�������:{data}', )
        self.socket.sendall(data)

    def close(self):
        self.socket.close()

class SocketManager:
    def __init__(self):
        self.socket = None
        self.lock = threading.Lock()
        self.connected = False

    def connect(self, server_address):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(server_address)
            self.connected = True
        except Exception as e:
            print(f"����ʧ��: {e}")
            self.socket = None
            self.connected = False

    def send0(self, data, index):
        with self.lock:  # ʹ����ȷ���̰߳�ȫ
            if not self.connected or self.socket is None:
                print("0Socketδ���ӻ��ѹرա�")
                return  # ���߿���ѡ����������������
            try:
                if self.socket is not None:
                    data_byte = bytes(data)
                    bytes_len = (1 + len(data_byte)).to_bytes(4, 'little')
                    message_to_send = bytes_len + (0).to_bytes(1, byteorder="little") + data_byte
                    self.socket.sendall(message_to_send)
                    print(f'{index}���͵�������:{message_to_send}', )
                    time.sleep(2)
            except Exception as e:
                print(f"{index}��������ʱ����: {e}")
                self.connected = False  # ��������쳣�����Ϊδ���� def send1(self, data, index):
    def send1(self, data, index):
        with self.lock:  # ʹ����ȷ���̰߳�ȫ
            if not self.connected or self.socket is None:
                print("1Socketδ���ӻ��ѹرա�")
                return  # ���߿���ѡ����������������
            try:
                if self.socket is not None:
                    data_byte = bytes(data)
                    bytes_len = (1 + len(data_byte)).to_bytes(4, 'little')
                    message_to_send = bytes_len + (1).to_bytes(1, byteorder="little") + data_byte
                    self.socket.sendall(message_to_send)
                    print(f'{index}���͵�������:{message_to_send}', )
                    time.sleep(1)
            except Exception as e:
                print(f"{index}��������ʱ����: {e}")
                self.connected = False  # ��������쳣�����Ϊδ����
    def send2(self, data, index):
        with self.lock:  # ʹ����ȷ���̰߳�ȫ
            if not self.connected or self.socket is None:
                print("2Socketδ���ӻ��ѹرա�")
                return  # ���߿���ѡ����������������
            try:
                selected_data = data[8:18]  # ������Ҫ����ѡȡ�ķ�Χ
                total_length = 1 + len(selected_data)
                bytes_len = total_length.to_bytes(4, 'little')
                message_to_send = bytes_len + (2).to_bytes(1, byteorder="little") + selected_data

                self.socket.sendall(message_to_send)
                print(f'{index}���͵�������:{message_to_send}', )
                time.sleep(1)
            except Exception as e:
                print(f"{index}��������ʱ����: {e}")
                self.connected = False  # ��������쳣�����Ϊδ����

    def close(self):
        if self.socket:
            self.socket.close()
            self.connected = False

# ��main�����г�ʼ��SocketManagerʵ��
socket_manager = SocketManager()
socket_manager.connect(('10.4.2.18', 30050))
def send_message_to_server_0(data):
    socket_manager.send0(data, 0)

def send_message_to_server_1(data):
    socket_manager.send1(data, 1)

def send_message_to_server_2(data):
    socket_manager.send2(data, 2)

# ���ӵ��������ĺ���
def connect_to_server():
    try:
        # ����һ��socket�����ӵ�������
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('10.4.2.18', 10086))  # �滻Ϊʵ�ʵķ�������ַ�Ͷ˿�
        print("�����ӵ�������")
        # global client_socket
        # client_socket = MySocket(s)

    except Exception as e:
        print(f"���ӵ�������ʱ����: {e}")

def send_message_to_server_qiti(verification_completed_event, shared_state):
    # time.sleep(3)
    # verification_completed_event.wait()
    # while True:
        # lock.acquire()
        # # ��λ��������ʾ��
        # try:
    read_data_command = bytes.fromhex('0A 03 04 11 10 00 00 0A 84 E3')  # ������CRCУ����
    read_data_command_with_crc = read_data_command
    ser.write(read_data_command_with_crc)
    read_data_2 = ser.read(20)  # ��ȡ20���ֽڵ�����
    # print(read_data_2)
    if len(read_data_2) == 20:
        print("Received data:", read_data_2.hex())
        data_bytes = bytes.fromhex(read_data_2.hex())
        print('2:', shared_state['plug_state'])
        if shared_state['plug_state'] == True:
            send_message_to_server_2(data_bytes)
                # time.sleep(5)
        # finally:
        #     lock.release()
        # ser.close()

def send_voice(data):
    try:
        # ����Socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # ���ӵ�������
        server_address = ('10.5.6.12', 10086)  # �滻Ϊʵ�ʵĵ�ַ�Ͷ˿�
        print("���ӵ� %s �˿� %s" % server_address)
        client_socket.connect(server_address)

        data_byte = data.encode()
        # print(data.encode())

        bytes_len = (1 + len(data_byte)).to_bytes(4, 'little')

        # message_to_send = bytes_len

        client_socket.sendall(bytes_len)
        client_socket.sendall((2).to_bytes(1, byteorder="little"))
        client_socket.sendall(data_byte)

        time.sleep(3)

    finally:
        # �ر�Socket
        print("�ر�����")
        client_socket.close()

def main():

    verification_completed_event = threading.Event()

    connect_to_server()

    shared_state = {'plug_state': True}
    global ser, plug
    plug = Device("192.168.1.107", "a9de4e0cfad666a9cd17e3d76f429dba")
    # ser = serial.Serial('COM3', 115200, timeout=1)  # ���ںš������ʡ���ʱʱ��

    # ����ѵ���õ�ģ��Ȩ��
    model_path = './best.pt'
    v5_path = 'E:/yolov5-slimming'
    choose_device = 'cuda:0'
    model = torch.hub.load(v5_path, 'custom', path=model_path, source='local', device=choose_device)

    camera = CameraSDK('10.4.2.24', 8000, 'admin', 'glkj@125')
    camera.start_thread()

    max_attempts = 9999  # ����Դ���
    attempts = 0

    # while attempts < max_attempts:
    #     if camera.image_data is not None:
    #         detector = cv2.QRCodeDetector()
    #         data, vertices_array, binary_qrcode = detector.detectAndDecode(camera.image_data)
    #
    #         if vertices_array is not None and data:
    #             send_voice('��Ʊ�ɹ����뽫������׼����ͷ����������֤')
    #             print("QRCode data:", data)
    #
    #             url = "http://10.5.6.11:6604/api/superviseOperator/operatorByCardno"
    #
    #             # ��������ͷ
    #             headers = {
    #                 'accept': 'text/plain',
    #                 'Content-Type': 'application/json',
    #                 'request-from': 'swagger'
    #             }
    #
    #             # ���������壨payload��
    #             payload = {
    #                 "cardNo": data
    #             }
    #
    #             try:
    #                 # ʹ��requests����POST����
    #                 response = requests.post(url, headers=headers, json=payload, timeout=10)
    #                 # ��������Ƿ�ɹ�
    #                 response.raise_for_status()
    #
    #                 response_data = response.json()
    #                 # print(response_data)
    #
    #                 # ����ɹ���������Ӧ
    #                 print("����ɹ���")
    #                 print(response_data['result'])
    #
    #                 if response_data['result'] != None:
    #
    #                     result = response_data['result']
    #                     base64_image_url = result['phoneUrl']
    #                     print(result['phoneUrl'])
    #                     # ʹ��requests��ȡͼƬ����
    #                     response = requests.get(base64_image_url)
    #                     print(response)
    #
    #                     # ȷ������ɹ�
    #                     if response.status_code == 200:
    #                         print('��ʼ����ʶ��')
    #                         app = FaceAnalysis(name='buffalo_l')  # ʹ�õļ��ģ����Ϊbuffalo_sc
    #                         app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_idС��0��ʾ��cpuԤ�⣬det_size��ʾresize���ͼƬ�ֱ���
    #
    #                         img = camera.image_data  # ��ȡͼƬ
    #                         faces = app.get(img)  # �õ�������Ϣ
    #
    #                         # ��������������ת��Ϊ����
    #                         feats = []
    #                         for face in faces:
    #                             feats.append(face.normed_embedding)
    #                         feats = np.array(feats, dtype=np.float32)
    #
    #                         # ��ȡĿ����������
    #                         target = np.frombuffer(response.content, dtype=np.uint8)
    #                         image = cv2.imdecode(target, cv2.IMREAD_COLOR)
    #                         target_faces = app.get(image)  # �õ�������Ϣ
    #                         # print(target_faces)
    #                         target_feat = np.array(target_faces[0].normed_embedding, dtype=np.float32)
    #
    #                         # �����������ƶȶԱ�
    #                         sims = np.dot(feats, target_feat)
    #                         print(sims)
    #                         # target_index = int(sims.argmax())
    #
    #                         data_0 = []
    #                         for i in range(2):
    #
    #                             if sims[i] > 0.4:
    #                                 send_voice('��֤�ɹ�����ҵ��ܿ�ʼ')
    #                                 data_0.append(1)
    #                                 if result['hasDocument'] == True:
    #                                     data_0.append(1)
    #                                 else:
    #                                     data_0.append(0)
    #                                 if shared_state['plug_state'] == True:
    #                                     send_message_to_server_0(data_0)
    #                                     verification_completed_event.set()
    #                                     break
    #                             else:
    #                                 send_voice('��Ա�������޷���ҵ��������ɨ����Ʊ')
    #                                 data_0.append(0)
    #                                 if result['hasDocument'] == True:
    #                                     data_0.append(1)
    #                                 else:
    #                                     data_0.append(0)
    #                         break
    #                 else:
    #                     print("δ�ҵ� 'result' �ֶλ��ֶ�ֵΪ��")
    #
    #             except requests.exceptions.HTTPError as errh:
    #                 print("Http Error:", errh)
    #             except requests.exceptions.ConnectionError as errc:
    #                 print("Error Connecting:", errc)
    #             except requests.exceptions.Timeout as errt:
    #                 print("Timeout Error:", errt)
    #             except requests.exceptions.RequestException as err:
    #                 print("OOps: Something Else", err)
    #
    #         else:
    #             print("��Ʊʧ�ܣ���������...")
    #             attempts += 1
    #     # else:
    #     #     print("û�м�⵽ͼ�����ݣ���������...")
    #
    #     if attempts == max_attempts:
    #         print("�ﵽ����Դ�����ֹͣ����")

    print("��ʼ")

    frame_processor = FrameProcessor(camera, model, verification_completed_event, shared_state)
    frame_processor_thread = threading.Thread(target=frame_processor.process)
    frame_processor_thread.daemon = True
    frame_processor_thread.start()

    # thread_qiti = threading.Thread(target=send_message_to_server_qiti, args=(verification_completed_event, shared_state))
    # thread_qiti.daemon = True
    # thread_qiti.start()

    # �ȴ��û�����q���˳�
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()



