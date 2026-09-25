import logging
import asyncio
from threading import Thread
from websockets.legacy.server import WebSocketServerProtocol
import websockets
import socket
logging.basicConfig(level=logging.INFO)

class WebSocketServer(object):
    def __init__(self,port):
        self.clients = set()
        self.loop = None
        self.thread = None
        self.port = port

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.run, args=(self.loop,))
        self.thread.start()

    def run(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def ws_handle(self, websocket: WebSocketServerProtocol, path):
        self.clients.add(websocket)
        logging.info(f'{websocket.remote_address} connects')
        try:
            async for message in websocket:
                await self.broadcast(message)
        except Exception as e:
            print(str(e))
        finally:
            self.clients.remove(websocket)

    async def sendmsg(self, websocket: WebSocketServerProtocol, msg):
        await websocket.send(msg)

    async def broadcast(self, msg):
        if self.clients:
            await asyncio.gather(*[self.sendmsg(ws, msg) for ws in self.clients])

    async def start_server(self):
        # hostname = socket.gethostname()
        # ip = socket.getaddrinfo(hostname, 8083)[0][4][0]
        server = await websockets.serve(self.ws_handle, '127.0.0.1', self.port, ping_interval=None)
        await server.wait_closed()

def main():
    ws_server = WebSocketServer()
    ws_server.start()
    try:
        loop = asyncio.get_event_loop()
        while True:
            # loop.run_until_complete(asyncio.sleep(1))
            loop.run_until_complete(ws_server.broadcast("1111"))
    except KeyboardInterrupt:
        ws_server.loop.stop()

# if __name__ == "__main__":
#     main()