from .core_analysis import (core_analysis)
from .re_encode import (re_encode)
from .convert_to_mono import (convert_to_mono)
from .convert_to_wav import (convert_to_wav)

class Transcriptor:

    def __init__(self, file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder=None, quantization=False):
        self.file = file
        self.voices_folder = voices_folder
        self.language = language
        self.log_folder = log_folder
        self.modelSize = modelSize
        self.quantization = quantization
        self.ACCESS_TOKEN = ACCESS_TOKEN
    def nemo(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "nemo", self.quantization)
        return res

    """     def whisper(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "whisper", self.quantization)
        return res
    

    
   def faster_whisper(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "faster-whisper", self.quantization)
        return res 

    def custom_whisper(self, custom_model_path):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "custom", self.quantization, custom_model_path)
        return res
    
    def huggingface_model(self, hf_model_id):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "huggingface", self.quantization, None, hf_model_id)
        return res
    
    def assemby_ai_model(self, aai_api_key):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "assemblyAI", self.quantization, None, None, aai_api_key)
        return res"""

class PreProcessor:
    '''
    class for preprocessing audio files.

    methods:

    re_encode(file) -> re-encode file to 16-bit PCM encoding  

    convert_to_mono(file) -> convert file from stereo to mono  

    mp3_to_wav(file) -> convert mp3 file to wav format  

    '''

    def re_encode(self, file):
        re_encode(file)
    
    def convert_to_mono(self, file):
        convert_to_mono(file)

    def convert_to_wav(self, file):
        path = convert_to_wav(file)
        return path
