import os
import platform
import socket
import sys
import struct
import numpy as np
import time

class ServiceWorker(object):
    def __init__(self):
        self.socket_name = 'my_socket_name'
        try:
            os.remove(self.socket_name)
        except OSError as e:
            if os.path.exists(self.socket_name):
                raise RuntimeError("socket already in use")
        socket_family = socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)

    def handle_connection(self, cl_socket):
        while True:
            size_pkt = cl_socket.recv(4)
            size_val, = struct.unpack('<I', size_pkt)
            print('size_val: ', size_val)

            data = bytearray()

            while len(data) < size_val:
                pkt = cl_socket.recv(size_val-len(data))
                if len(pkt) == 0:
                    print('Frontend disconnected.')
                    sys.exit(0)
                data += pkt
            pkt = data

            print(f"Received {len(pkt)} bytes")
            #resp = list()
            #for b in pkt:
            #    resp.append(b + 1)
            #resp_bytes = bytes(resp)
            now = time.time()
            y = np.frombuffer(pkt, dtype='uint8').reshape((40,3,1024,1024))
            y = y + 1
            resp_bytes = y.tobytes()
            t = time.time() - now
            print('processing end, time:', t)
            print('sending size_val...')
            cl_socket.sendall(struct.pack("<I", size_val))
            print(f'sending resp_bytes {len(resp_bytes)}...')
            cl_socket.sendall(resp_bytes)
            print(f"Sent {size_val} bytes back.")
    
    
    def run_server(self):
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.socket_name)
        self.sock.listen(1)

        while True:
            print('Waiting for connections...')
            (cl_socket, _) = self.sock.accept()
            cl_socket.setblocking(True)
            print(f'Connection accepted: {cl_socket.getsockname()}')
            self.handle_connection(cl_socket)

worker = ServiceWorker()
worker.run_server()
