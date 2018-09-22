import socket

s = socket.socket()
host = socket.gethostname()
port = 12222

s.connect((host, port))
# your text is here
some_text = 'I am having a good time'
s.send(some_text.encode('utf-8'))
print(s.recv(1024).decode('utf-8'))
