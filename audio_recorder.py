import pyaudiowpatch as pyaudio
import wave

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import torchaudio

import asyncio

class Recorder: 
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 512
        self.NO_OF_CHUNK = 30 # ~30 = 1 second, 512 * 30 ~= 16000 which is 1s
        self.TARGET_SAMPLING_RATE = 16000
        self.speaker_sampling_rate = 0
        self.scale_factor = 0
        self.speaker_channel = 0
        self.default_speakers = None
        self.audio_interface = pyaudio.PyAudio()
        self.audio_format = pyaudio.paInt16
    
    def set_callback_fn(self, callback_fn):
        self.callback_fn = callback_fn # pass the data for socket send

    def verify_loopback_setup(self):
        try:
            # Get default WASAPI info
            wasapi_info = self.audio_interface.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("WASAPI is not available on the system")
            return False

        default_speakers = self.audio_interface.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not default_speakers["isLoopbackDevice"]:
            for loopback in self.audio_interface.get_loopback_device_info_generator():
                """
                Try to find loopback device with same name(and [Loopback suffix]).
                Unfortunately, this is the most adequate way at the moment.
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    print(f"the default loopback speaker {loopback}")
                    break
            else:
                print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                return False
        
        self.speaker = default_speakers
        self.speaker_channel = self.speaker["maxInputChannels"]
        return True

    def lookup_sampling_rate(self):
        list_available_rate = [16000, 48000, 24000]
        # list_available_rate = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        print(f"check format: index {self.speaker.get('index')}, channel {self.speaker.get('maxInputChannels')}, format {self.audio_format}")
        for rate in list_available_rate:
            try:
                available = self.audio_interface.is_format_supported(
                    rate,
                    input_device=self.speaker.get('index'),  # Changed to input_device
                    input_channels=self.speaker.get('maxInputChannels'),  # Changed to input_channels
                    input_format=self.audio_format  # Changed to input_format
                    )
                if available:
                    self.speaker_sampling_rate = rate
                    self.scale_factor = rate / self.TARGET_SAMPLING_RATE
                    print(f"sampling rate selected: {rate}")
                    return True
            except:
                print(f"{rate} is not supported, check next rate")
                continue
        return False

    def record_start(self):
        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            if self.scale_factor == 1:
                if self.speaker_channel > 1:
                    ndarray=self.buffer_to_nparray(in_data, self.speaker_channel)
                    ndarray = self.merge_channel(ndarray)
                    self.callback_fn(self.nparray_to_buffer(ndarray, self.speaker_channel))
                else:
                    self.callback_fn(in_data)
            else:
                #may need to add low pass filter if found aliasing impact too much
                ndarray=self.buffer_to_nparray(in_data, self.speaker_channel)
                ndarray = self.lowpass_filter(ndarray, self.TARGET_SAMPLING_RATE / 2, self.speaker_sampling_rate)
                resampled = resample_poly(ndarray, self.TARGET_SAMPLING_RATE, self.speaker_sampling_rate, axis=0)
                if self.speaker_channel > 1:
                    resampled = self.merge_channel(resampled)
                self.callback_fn(self.nparray_to_buffer(resampled, self.speaker_channel))
            return (in_data, pyaudio.paContinue)
        self.stream = self.audio_interface.open(format=pyaudio.paInt16,
            channels=self.speaker_channel,
            rate=int(self.speaker_sampling_rate),
            frames_per_buffer=int(self.CHUNK_SIZE * self.scale_factor * self.NO_OF_CHUNK),
            input=True,
            input_device_index=self.speaker["index"],
            stream_callback=callback
        ) 

    def record_stop(self):
        """Clean up audio resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Error cleaning up audio resources: {e}")
    
    def is_recording(self):
        print(f"{self.stream}, {self.callback_fn}")

    def cleanup(self):
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
        
    def buffer_to_nparray(self, data, channel):
        try:
            assert channel == 1 or channel == 2
        except AssertionError:
            raise Exception("invalid number of channel")
        
        frame = np.frombuffer(data, dtype=np.int16)  
        if channel == 2:
            # interleaved channels
            frame = frame.reshape(-1, 2).astype(np.int16)
            # frame = np.stack((frame[::2], frame[1::2]), axis=0)  # channels on separate axes
        return frame
    
    def nparray_to_buffer(self, data, channel):
        try:
            assert channel == 1 or channel == 2
        except AssertionError:
            raise Exception("invalid number of channel")
        if channel == 2:
            return data.flatten().astype(np.int16).tobytes() #https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.flatten.html
        else:
            return data.flatten().astype(np.int16).tobytes()

    def merge_channel(self, data):
        return np.squeeze(np.mean(data, axis=1).astype(np.int16))

    def lowpass_filter(self, signal, cutoff_freq, sample_rate):
        """
        Apply a low-pass Butterworth filter to prevent aliasing in the signal.

        Args:
            signal (np.ndarray): Input audio signal to filter
            cutoff_freq (float): Cutoff frequency in Hz
            sample_rate (float): Sampling rate of the input signal in Hz

        Returns:
            np.ndarray: Filtered audio signal

        Notes:
            - Uses a 5th order Butterworth filter
            - Applies zero-phase filtering using filtfilt
        """
        # Calculate the Nyquist frequency (half the sample rate)
        nyquist_rate = sample_rate / 2.0

        # Normalize cutoff frequency to Nyquist rate (required by butter())
        normal_cutoff = cutoff_freq / nyquist_rate

        # Design the Butterworth filter
        b, a = butter(5, normal_cutoff, btype='low', analog=False)

        # Apply zero-phase filtering (forward and backward)
        # fdata = scipy.signal.filtfilt(b, a, np.frombuffer(data), axis=0).tobytes()
    # Apply zero-phase filter to each channel
        for channel in range(signal.shape[1]):
            signal[:, channel] = filtfilt(b, a, signal[:, channel])
        # filtered_signal = filtfilt(b, a, signal, axis=0).tobytes() # require filter for each channel when use
        return signal

async def record(rec: Recorder, DURATION=10):
    print("async record")
    filename = "loopback_record.wav"

    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(1) 
    wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(rec.TARGET_SAMPLING_RATE)
    rec.set_callback_fn(lambda data: wave_file.writeframes(data))
    rec.record_start()
    await asyncio.sleep(DURATION)
    rec.record_stop()

if __name__ == "__main__":
    rec = Recorder()
    loopback_device_result = rec.verify_loopback_setup()
    print(f"is loopback device avabilable: {loopback_device_result}")

    if loopback_device_result:
        rec.lookup_sampling_rate()
        print(f"the best sample rate is {rec.speaker_sampling_rate}")
        print(f"the scale of sampling rate to target rate 16000 is {rec.scale_factor}")

        if not rec.speaker_sampling_rate == 0:
            asyncio.run(record(rec))
            audio_file = "loopback_record.wav"
            origin_metadata = torchaudio.info(audio_file)
            print(f"original audio file info {origin_metadata}")


