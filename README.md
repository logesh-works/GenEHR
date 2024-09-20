### Requirements

* Python 3.8 or greater

### GPU execution

GPU execution needs CUDA 11.  

GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 11](https://developer.nvidia.com/cublas)
* [cuDNN 8 for CUDA 11](https://developer.nvidia.com/cudnn)

There are multiple ways to install these libraries. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below.

### Google Colab:

on google colab run this to install CUDA dependencies:
```
!apt install libcublas11
```


### installation:
```
!apt install libcublas11
```
```
pip install -r requirments.txt
```
### To Run:

```
python -m GenEHR
```

 Does speaker diarization, speaker recognition, and transcription on a single wav file to provide a transcript with actual speaker names. This library will also return an array containing result information. ⚙ 

This library contains following audio preprocessing functions:

1. convert other audio formats to wav

2. convert stereo wav file to mono

3. re-encode the wav file to have 16-bit PCM encoding

Transcriptor method takes 7 arguments. 

1. file to transcribe

2. log_folder to store transcription

3. language used for transcribing (language code is used)

4. model size ("tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3")

5. ACCESS_TOKEN: huggingface acccess token (also get permission to access `pyannote/speaker-diarization@2.1`)

6. voices_folder (contains speaker voice samples for speaker recognition)

7. quantization: this determine whether to use int8 quantization or not. Quantization may speed up the process but lower the accuracy.

voices_folder should contain subfolders named with speaker names. Each subfolder belongs to a speaker and it can contain many voice samples. This will be used for speaker recognition to identify the speaker.

if voices_folder is not provided then speaker tags will be arbitrary.

log_folder is to store the final transcript as a text file.

transcript will also indicate the timeframe in seconds where each speaker speaks.

### Transcription example:

```
Change this in main.py

# use normal whisper
res = transcriptor.whisper()

# use faster-whisper (simply faster)
res = transcriptor.faster_whisper()

# use a custom trained whisper model
res = transcriptor.custom_whisper("D:/whisper_tiny_model/tiny.pt")

# use a huggingface whisper model
res = transcriptor.huggingface_model("Jingmiao/whisper-small-chinese_base")

# use assembly ai model
res = transcriptor.assemby_ai_model("your api key")

res --> [["start", "end", "text", "speaker"], ["start", "end", "text", "speaker"]...]
```

#### if you don't want speaker names: keep voices_folder as an empty string ""

start: starting time of speech in seconds  
end: ending time of speech in seconds  
text: transcribed text for speech during start and end  
speaker: speaker of the text 



#### File and its purposs:

1. Convert_to_mono.py
        This is file is used for convert stereo audio to mono audio
2. convert_to_wav.py
        This file is used for convert other formate audio file to wav audio file
3. re_encode.py
        This file is used for encode the audio into 16 bit PCM
4. wav_segmenter.py
        Converting full audio to spearate segment to transcribe



### Performance
```
These metrics are from Google Colab tests.
These metrics do not take into account model download times.
These metrics are done without quantization enabled.
(quantization will make this even faster)

metrics for faster-whisper "tiny" model:
    on gpu:
        audio name: obama_zach.wav
        duration: 6 min 36 s
        diarization time: 24s
        speaker recognition time: 10s
        transcription time: 64s


metrics for faster-whisper "small" model:
    on gpu:
        audio name: obama_zach.wav
        duration: 6 min 36 s
        diarization time: 24s
        speaker recognition time: 10s
        transcription time: 95s


metrics for faster-whisper "medium" model:
    on gpu:
        audio name: obama_zach.wav
        duration: 6 min 36 s
        diarization time: 24s
        speaker recognition time: 10s
        transcription time: 193s


metrics for faster-whisper "large" model:
    on gpu:
        audio name: obama_zach.wav
        duration: 6 min 36 s
        diarization time: 24s
        speaker recognition time: 10s
        transcription time: 343s
```

#### why not using pyannote/speaker-diarization-3.1, speechbrain >= 1.0.0, faster-whisper >= 1.0.0:

because older versions give more accurate transcriptions. this was tested.

This library uses following huggingface models:

#### https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
#### https://huggingface.co/pyannote/speaker-diarization