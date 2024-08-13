import socket
import numpy as np
import time
import struct

class Interface :
    
    def __init__(self, send_port, id_ = "@"):
        self.id_ = id_
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_port = send_port
    
    def send_parser(self, message, type_mes) :
        if type_mes == "vel_qual" :
            buffer = bytes(self.id_, encoding = "ascii")
            vel = message[0]
            qual = message[1]
            for i in vel :
                buffer += struct.pack("f", i)
            buffer += struct.pack("B", qual)
        return buffer
    
    def send(self, message, type_mes):
        buffer = self.send_parser(message, type_mes)
        self.send_socket.sendto(buffer, ("192.168.2.2", self.send_port))
        

