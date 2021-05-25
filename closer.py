from multiprocessing.connection import Client

address = ('localhost', 6000)
conn = Client(address, authkey=str.encode('secret password'))
conn.send('close')
conn.close()