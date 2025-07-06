import asyncio
import websockets
import logging
from typing import Optional
import numpy as np
import sys
import queue

from asr_engine import vac_factory
from asr_engine import AudioTranscriber

MODEL_SIZE = "medium"

class WebSocketServer:
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.client: Optional[websockets.WebSocketServerProtocol] = None
        self.server: Optional[websockets.WebSocketServer] = None
        self.logger = logging.getLogger(__name__)

    async def _handler(self, websocket: websockets.WebSocketServerProtocol):
        """Handle incoming WebSocket connections."""
        if self.client is not None:
            await websocket.send("Server busy: Only one connection allowed")
            await websocket.close()
            return

        self.client = websocket
        self.on_connect(websocket)
        try:
            async for message in websocket:
                self.on_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.client = None
            self.on_disconnect(websocket)

    async def send_message(self, message: str):
        """Send a message to the connected client."""
        if self.client is not None:
            await self.client.send(message)

    def on_connect(self, client: websockets.WebSocketServerProtocol):
        """Called when a new client connects."""
        self.logger.info(f"Client connected: {client.remote_address}")


    def on_message(self, client: websockets.WebSocketServerProtocol, message: str):
        """Called when a message is received from the client."""
        # self.logger.info(f"Received message from {client.remote_address}: {message}")
        # self.logger.info(f"Received message from {client.remote_address}: {type(message)}")

    def on_disconnect(self, client: websockets.WebSocketServerProtocol):
        """Called when the client disconnects."""
        self.logger.info(f"Client disconnected: {client.remote_address}")

    async def start(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(self._handler, self.host, self.port)
        self.logger.info(f"Server started on ws://{self.host}:{self.port}")
        
        await self.server.wait_closed()

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("Server stopped")

class SingleClientWebSocketServer(WebSocketServer):
    def on_message(self, client, message):
        super().on_message(client, message)
        if isinstance(message, bytes):
            self.data_buffer.put(message)
            cur_size = len(message) * self.data_buffer.qsize()
            if cur_size >= self.minlimit:
                # the input data is 16000 hz mono buffer data, convert to np array for upcoming process
                # i.e. vac and whisper
                combined_signal = self.buffer_to_nparray(self.concat_bytes_from_queue(self.data_buffer)) #check buffer size

                if not self.vac_mode:
                    output_message = self.audio_transcriber.handle_audio_chunk(combined_signal)
                    print(f"output_message: {output_message}")
                    try:
                        if output_message is not None:
                            asyncio.create_task(self.send_message(f"{output_message}"))
                    except BrokenPipeError:
                        self.logger.info("broken pipe -- connection closed?")
                else:
                    res = self.vac(combined_signal)
                    print(f"vac result: {res}")


    def __init__(self, min_second, audio_transcriber, vac, vac_mode=False):
        super().__init__()
        SAMPLING_RATE = 16000
        BYTE_SIZE = 2
        self.last_end = None
        self.is_first = True
        self.minlimit = min_second * SAMPLING_RATE * BYTE_SIZE
        self.data_buffer = queue.Queue()
        self.logger = logging.getLogger(__name__)
        self.audio_transcriber = audio_transcriber
        self.vac = vac
        self.vac_mode = vac_mode

    def on_disconnect(self, client: websockets.WebSocketServerProtocol):
        self.data_buffer = queue.Queue()
        self.audio_transcriber.clear_buffer()
        super().on_disconnect(client)

    def on_connect(self, client):
        super().on_connect(client)

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            self.logger.debug("No text in this segment")
            return None


    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.send_message(msg)

    def concat_bytes_from_queue(self, byte_queue):
        byte_list = []
        while not byte_queue.empty():
            try:
                byte_list.append(byte_queue.get_nowait())
            except queue.Empty:
                break
        return b"".join(byte_list)
    
    def buffer_to_nparray(self, data):
        return np.astype(np.frombuffer(data, dtype=np.int16) / 32768.0, np.float32) #divided by 32768.0 to make the buffer `works` in whisper
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    min_chunk = 1
    size = MODEL_SIZE #model
    vac = vac_factory()
    audio_transcriber = AudioTranscriber(asr_model_size=size)
    audio_transcriber.warmup_wshiper('loopback_record.wav')
    audio_transcriber.set_language_code(language_code='auto')
    server = SingleClientWebSocketServer(min_chunk, audio_transcriber, vac, vac_mode=False)

    try:
        asyncio.get_event_loop().run_until_complete(server.start())
    except KeyboardInterrupt:
        asyncio.get_event_loop().run_until_complete(server.stop())