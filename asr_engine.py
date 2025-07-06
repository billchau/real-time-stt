import numpy as np
import sys
import logging
from wtpsplit import WtP
logger = logging.getLogger(__name__)
from wtpsplit import WtP

WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

class FasterWhisperASR():
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """
    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}

        from faster_whisper import WhisperModel
#        logging.getLogger("faster_whisper").setLevel(logger.level)

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        print(f"model_size_or_path {model_size_or_path}")
        # self.model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        self.model = WhisperModel(model_size_or_path, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
#        self.model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")


    def transcribe(self, audio, init_prompt="", language_code=None):
        if language_code is not None and language_code != 'auto':
            self.transcribe_kargs['language'] = language_code
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        # print(f"detected lan {info}")  # info contains language detection result

        return list(segments), info

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

def vac_factory():
    import torch
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    from silero_vad_iterator import FixedVADIterator
    vac = FixedVADIterator(model)  # we use the default options there: 500ms silence, 100ms padding, etc.  
    vac.reset_states()
    return vac

def get_separator(sep=' ', language_code=None):
    if language_code is not None:
        if language_code in ['ja', 'kr', 'yue', 'zh']:
            separator = ''
        else:
            separator = ' '
    else: 
        separator = sep
    return separator

class WtPtok:
    def __init__(self):
        self.wtp = WtP("wtp-canine-s-12l-no-adapters")

    def split(self, message, lang_code):
        return self.wtp.split(message, lang_code=lang_code)
    
class WordBuffer:
    def __init__(self):
        self.buffer_prompt = []
        self.commit_prompt = []

    def add_word(self, word):
        self.buffer_prompt.append(word)  # Add new word

    def add_words(self, words):
        self.buffer_prompt.append(words)

    def concatenate(self, is_commit_all=False, sep=' ', language_code=None):
        if language_code is not None:
            if language_code in ['ja', 'kr', 'yue', 'zh']:
                separator = ''
            else:
                separator = ' '
        else: 
            separator = sep
        return  separator.join(self.commit_prompt) if is_commit_all else separator.join(self.commit_prompt) + separator.join(self.buffer_prompt)

    def commit(self, commit_prompt):
        self.commit_prompt = list(commit_prompt)
        self.buffer_prompt = []

    def clear(self):
        self.buffer_prompt = []
        self.commit_prompt = []

    def display_prompt(self):
        print(f"prompt_commit: {''.join(self.commit_prompt)}")
        print(f"prompt_buffer: {''.join(self.buffer_prompt)}")
    
class LanguageCounter:
    def __init__(self, min_count=3, force_language_code=None):
        self.force_language_code = force_language_code 
        self.min_count = min_count
        self._counts = {}
    
    def set_language_code(self, language_code):
        self.force_language_code = language_code

    def add_language(self, *language_codes):
        for s in language_codes:
            self._counts[s] = self._counts.get(s, 0) + 1
    
    def get_max_count(self):
        return max(self._counts.values()) if self._counts else 0
    
    def get_language_code(self):
        if self.force_language_code is not None:
            return self.force_language_code
        
        if not self._counts:
            return None
        
        max_count = self.get_max_count()
        if max_count < self.min_count:
            return None
        
        for language_code, count in self._counts.items():
            if count == max_count:
                return language_code
    
    def clear(self):
        self._counts.clear()

class AudioBuffer:
    def __init__(self, max_buffer = 5):
        self.buffer = []
        self.max_buffer = max_buffer

    def add_audio_buffer(self, buffer):
        self.buffer.append(buffer) 
        print(f"buffer size : {len(self.buffer)}")

    def get_buffer(self):
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.buffer, axis=0)

    def reach_max_buffer(self):
        return len(self.buffer) == self.max_buffer

    def clear(self):
        self.buffer = []

class AudioTranscriber():
    def __init__(self, asr_model_size='base', language_code=None):
        self.tokenizer = WtPtok()

        # downloads the model from huggingface on the first use
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)  # we use the default options there: 500ms silence, 100ms padding, etc.  
        self.vac.reset_states()
        self.NO_VOICE_STATUS  = 'no_voice'
        self.VOICE_STATUS  = 'voice'
        self.status = self.NO_VOICE_STATUS
        self.offset = 0
        self.prompt_buffer = WordBuffer()
        self.audio_buffer = AudioBuffer()
        self.language_counter = LanguageCounter(force_language_code=language_code)
        self.asr = FasterWhisperASR(modelsize=asr_model_size)

    def set_language_code(self, language_code):
        self.language_counter.set_language_code(language_code)

    def transcribe(self, audio_chunk, commit_all):
        
        self.audio_buffer.add_audio_buffer(audio_chunk)
        print(f"commit_all {commit_all} or max buffer {self.audio_buffer.reach_max_buffer()}")
        final_commit_decision = self.audio_buffer.reach_max_buffer() or commit_all
        
        prompt = self.prompt_buffer.concatenate(is_commit_all=final_commit_decision, language_code=self.language_counter.get_language_code())
        print(f"reach max buffer size, force commit? {final_commit_decision}")
        if final_commit_decision:
            chunk_for_transcribe = self.audio_buffer.get_buffer()
            print(f"use buffer audio {chunk_for_transcribe.shape}")
            self.audio_buffer.clear()
        else:
            chunk_for_transcribe = audio_chunk
            print(f"use chunk")
        print(f"processing chunk {chunk_for_transcribe.shape}")
        segments, info = self.asr.transcribe(chunk_for_transcribe, init_prompt=prompt, language_code=self.language_counter.get_language_code())
        self.language_counter.add_language(info.language)
        output_message = self.handle_transcript(segments, final_commit_decision)
        # output message format{'commited': 'msg', 'buffer': 'msg'} or {}

        self.prompt_buffer.display_prompt()

        return output_message, info.language

    def handle_transcript(self, segments, commit_all):
        output_message = ''
        words = self.ts_words(segments=segments)
        msg = "".join(str(t[2]) for t in words)
        self.buffer = words  # Add new word [Transcript1, Transcript2]
        if commit_all:
            output_message = {"commit": msg, "buffer": ""}
        else:
            output_message = {"commit": "", "buffer": msg}

        if commit_all:
            self.prompt_buffer.commit(msg)
        else:
            self.prompt_buffer.add_words(msg)
        # output_message = get_separator(language_code=self.language_counter.get_language_code()).join(o)
        # output message format{'commited': 'msg', 'buffer': 'msg'} or {}

        return output_message

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o
        
    def handle_audio_chunk(self, audio_chunk):
        #vad result format
        #{'start': 2147776}
        #None  <-- in between voice and no voice, status unchange
        #{'end': 1900096}
        #{'start': 283072, 'end': 289344}
        res = self.vac(audio_chunk)
        print(f"Vac result: {res}")
        commit_all = False
        if self.status == self.VOICE_STATUS:
            
            if res is not None and 'end' in res:
                if 'start' not in res:
                    self.status = self.NO_VOICE_STATUS
                    self.clear_buffer()
                commit_all = True
            output_message, lanuage = self.transcribe(audio_chunk, commit_all)
            return output_message
        else:
            if res is not None and 'start' in res:
                if 'end' not in res:
                    self.status = self.VOICE_STATUS
                if 'end' in res:
                    commit_all = True
                output_message, lanuage = self.transcribe(audio_chunk, commit_all)
                return output_message
                #add full stop?
        return {}
    
    def clear_buffer(self):
        self.vac.reset_states()
        self.status = self.NO_VOICE_STATUS

    def warmup_wshiper(self, wav_file):
        try:
            segments, info = self.asr.transcribe(wav_file, init_prompt='')
            print(f"warm up completed")
        except Exception as e:
            print(str(e))
            print(f"failed to warnup")
        