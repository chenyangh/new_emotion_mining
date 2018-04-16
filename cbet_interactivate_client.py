import socket

s = socket.socket()
host = socket.gethostname()
port = 12221

s.connect((host, port))
# your text is here
some_text = 'happy'
s.send(some_text.encode('utf-8'))
print(s.recv(1024).decode('utf-8'))
