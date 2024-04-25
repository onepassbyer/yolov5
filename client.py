import socket
import threading

# 定义处理连接的函数
def handle_connection(conn):
    try:
        while True:
            data = conn.recv(4096)  # 接收数据（最大字节）
            if not data:
                break  # 如果没有数据，跳出循环
            message = data.decode()  # 解码收到的数据
            print("接收到的数据:", message)
            f = open("message1.txt",'a')
            f.write(message)
            f.write('\n')
            f.close()
    except Exception as e:
        print("处理连接时发生错误:", e)
    finally:
        conn.close()  # 关闭连接
# 创建Socket对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 指定IP地址和端口号
host = '127.0.0.1'  # 监听所有接口
port = 54774
# 绑定Socket到地址和端口
server_socket.bind((host, port))
# 开始监听连接
server_socket.listen(1)
print("等待连接...")
try:
    while True:
        # 等待连接请求
        conn, addr = server_socket.accept()
        # print('连接来自:', addr)
        # 创建并启动新线程处理连接
        thread = threading.Thread(target=handle_connection, args=(conn,))
        thread.start()
except KeyboardInterrupt:
    print('手动关闭连接')
    server_socket.close()
