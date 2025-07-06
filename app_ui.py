
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QTextEdit
    
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtCore import pyqtSignal, QObject
from audio_recorder import Recorder
import qasync
import asyncio
from websockets import connect  # WebSocket client library
import time
import ast

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
FONT_SIZE = 14
FONT_FAMILY = "Arial"
TRANSCRIPTION_AREA_HEIGHT = 600
SERVER_URL = "ws://localhost:8765"


STATUS = [
    "idle"
    "recording"
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("loopback audio transcription tool created by billchau")
        self.resize(WINDOW_HEIGHT, WINDOW_WIDTH)

        default_font = QFont(FONT_FAMILY, FONT_SIZE)
        #params
        self.language_detected = "N/A"
        self.status = [0]
        self.ws_client = None
        self.main_layout = QVBoxLayout()
    
        #translation output
        self.display_layout = QVBoxLayout()
        self.display_transcription_label = QLabel("Transcription")
        self.display_transcription_label.setFont(default_font)
        self.display_transcription_textarea = QTextEdit("")
        self.display_transcription_textarea.setReadOnly(True)
        self.display_transcription_textarea.setFont(default_font)

        self.display_layout.addWidget(self.display_transcription_label)
        self.display_layout.addWidget(self.display_transcription_textarea)
        self.main_layout.addLayout(self.display_layout)

        self.message_label = QLabel()
        self.message_label.setFont(default_font)
        self.main_layout.addWidget(self.message_label)

        self.display_control_layout = QHBoxLayout()
        self.clear_display_button = QPushButton("Clear display")
        self.clear_display_button.setFont(default_font)
        self.clear_display_button.clicked.connect(self.reset_display)
        self.test_connection_button = QPushButton("Test connecton")
        self.test_connection_button.setFont(default_font)
        self.test_connection_button.clicked.connect(self.test_connection)
        self.test_connection_button.setEnabled(False)
        self.save_as_button = QPushButton("Save output as")
        self.save_as_button.setFont(default_font)
        self.save_as_button.clicked.connect(self.save_text)

        self.display_control_layout.addWidget(self.save_as_button)
        self.display_control_layout.addWidget(self.clear_display_button)
        self.display_control_layout.addWidget(self.test_connection_button)
        self.main_layout.addLayout(self.display_control_layout)

        #control
        self.control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setFont(default_font)
        self.start_button.clicked.connect(self.start_transcript)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFont(default_font)
        self.stop_button.clicked.connect(self.stop_transcript)
        self.stop_button.setEnabled(False)

        self.control_layout.addWidget(self.start_button)
        self.control_layout.addWidget(self.stop_button)

        self.main_layout.addLayout(self.control_layout)

        
        self.widget = QWidget()
        self.widget.setLayout(self.main_layout)
        self.setCentralWidget(self.widget)

        self.clear_display_track()

    def start_transcript(self):
        self.ws_client = WebSocketClient(SERVER_URL)
        self.ws_client.message_received.connect(
            lambda msg: self.handle_server_msg(msg)
        )
        self.ws_client.connection_closed.connect(
            lambda: self.update_message("Disconnected from server")
        )
        self.ws_client.error_occurred.connect(
            lambda msg: self.update_message(f"Error: {msg}")
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        asyncio.create_task(self.ws_client.connect())
        # #disable layout operation
        self.change_layout_mode(False)
        self.loop = asyncio.get_running_loop()

        self.rec = Recorder()
        loopback_device_result = self.rec.verify_loopback_setup()
        print(f"is loopback device avabilable: {loopback_device_result}")
        if loopback_device_result:
            self.rec.lookup_sampling_rate()
            print(f"the best sample rate is {self.rec.speaker_sampling_rate}")
            print(f"the scale of sampling rate to target rate 16000 is {self.rec.scale_factor}")

            if not self.rec.speaker_sampling_rate == 0:
                self.rec.set_callback_fn(
                    lambda in_data: self.loop.call_soon_threadsafe(self.ws_client.audio_queue.put_nowait, in_data)
                )
                self.rec.record_start()
                asyncio.create_task(self.ws_client.send_audio_signal())
            else:
                self.update_message("Sampling rate not supported")
                self.stop_transcript()
        else:
            self.update_message("Loopback device not set")
            self.stop_transcript()

    def stop_transcript(self):
        print("stop")
        # enable buttons and clear cache
        if self.ws_client:
            asyncio.create_task(self.ws_client.disconnect())
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        self.rec.record_stop()
        self.change_layout_mode(True)


    def test_connection(self):
        if self.ws_client:
            asyncio.create_task(self.ws_client.test_connection())
            self.update_message(f"Test connection")
            
    #enable
    def change_layout_mode(self, is_idle):
        print("change_layout_mode")
        self.test_connection_button.setEnabled(not is_idle)

    def reset_display(self):
        print("reset display")
        self.display_transcription_textarea.setText("")
        self.clear_display_track()

    def update_message(self, message):
        print("update_message " + message)
        self.message_label.setText(message)

    def handle_server_msg(self, message):
        print(f"message: {message}")
        message_obj = ast.literal_eval(message)
        if message_obj and message_obj != {}:
            buffer_message = message_obj['buffer']
            commit_message = message_obj['commit']
            if commit_message and len(commit_message) > 0:
                self.add_message('commit', commit_message)
            if buffer_message and len(buffer_message) > 0:
                self.add_message('buffer', buffer_message)

    def save_text(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.display_transcription_textarea.toPlainText())
    
    def add_message(self, msg_type, message):
        if not message.strip():
            return
            
        cursor = self.display_transcription_textarea.textCursor()
        format_normal = QTextCharFormat()
        
        if msg_type == "buffer":
            # Remove previous buffer if exists
            if not self.has_pending_buffer:
                # Move to end of document
                cursor.movePosition(QTextCursor.End)
                self.buffer_start_pos = cursor.position()
            
            # Create buffer formatting (gray and italic)
            buffer_format = QTextCharFormat()
            buffer_format.setForeground(QColor("gray"))
            buffer_format.setFontItalic(True)
            
            # Insert buffer message without line break
            cursor.insertText(message, buffer_format)
            self.buffer_end_pos = cursor.position()
            self.has_pending_buffer = True
            
        elif msg_type == "commit":
            if self.has_pending_buffer:
                # Remove previous buffer
                cursor.setPosition(self.buffer_start_pos)
                cursor.setPosition(self.buffer_end_pos, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                
                # Insert commit message at buffer position
                cursor.insertText(message, format_normal)
                self.has_pending_buffer = False
            else:
                # Append commit message normally
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(message, format_normal)
            
            # Check if message ends with newline to start new line
            if not message.endswith('\n'):
                # Move to end to prepare for next buffer
                cursor.movePosition(QTextCursor.End)
        
        # Clear input and scroll to bottom
        self.display_transcription_textarea.ensureCursorVisible()

    def clear_display_track(self):
        self.buffer_start_pos = -1
        self.buffer_end_pos = -1
        self.has_pending_buffer = False
 
# Async WebSocket Client
class WebSocketClient(QObject):
    message_received = pyqtSignal(str)
    connection_closed = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.websocket = None
        self.running = False
        self.audio_queue = asyncio.Queue(maxsize=20)

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await connect(self.url)
            self.running = True
            asyncio.create_task(self.listen())
        except Exception as e:
            self.error_occurred.emit(str(e))

    async def disconnect(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.running = False
            self.connection_closed.emit()

    async def listen(self):
        """Listen for incoming messages"""
        while self.running:
            try:
                message = await self.websocket.recv()
                self.message_received.emit(message)
            except Exception as e:
                self.error_occurred.emit(str(e))
                self.running = False

    async def send_message(self, message):
        """Send a message to the server"""
        if self.websocket and self.running:
            await self.websocket.send(message)
            await asyncio.sleep(0.02)

    async def send_audio_signal(self):
        """Send a message to the server"""
        print("try to send msg")
        try:
            while True:
                in_data = await self.audio_queue.get()
                await self.websocket.send(in_data)
                print(f"data sent at {time.time()}")
                await asyncio.sleep(0.02)
        except self.websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")

    async def test_connection(self):
        """Send a message to the server"""
        if self.websocket and self.running:
            await self.websocket.send("Connection completed")


if __name__ == "__main__":
    app = QApplication([])
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow()
    window.show()
    with loop:
        loop.run_forever()