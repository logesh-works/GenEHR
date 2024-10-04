def transcribe(file,model):
    res = ""  # NeMo transcription logic
    try:
        res = model.transcribe([file], batch_size=1,logprobs=False, language_id='ta')[0]  # Returns a list of transcriptions
        return res
    except Exception as err:
        raise Exception(f"an error occurred while transcribing with NeMo: {err}")
   



"""
----------------------------------------------------------------------------
import torch
from faster_whisper import WhisperModel
import whisper
import os
from transformers import pipeline
import assemblyai as aai
import soundfile as sf
import nemo.collections.asr as nemo_asr  # Import NeMo ASR

def transcribe(file, language, model_size, model_type, quantization, custom_model_path, hf_model_path, aai_api_key):
    res = ""
    if model_size in ["base", "tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
        if model_type == "faster-whisper":
            if torch.cuda.is_available():
                if quantization:
                    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                else:
                    model = WhisperModel(model_size, device="cuda", compute_type="float16")
            else:
                if quantization:
                    model = WhisperModel(model_size, device="cpu", compute_type="int8")
                else:
                    model = WhisperModel(model_size, device="cpu", compute_type="float32")

            if language in model.supported_languages:
                segments, info = model.transcribe(file, language=language, beam_size=5)

                for segment in segments:
                    res += segment.text + " "
                    
                return res
            else:
                Exception("Language code not supported.\nThese are the supported languages:\n", model.supported_languages)
        elif model_type == "whisper":
            try:
                if torch.cuda.is_available():
                    model = whisper.load_model(model_size, device="cuda")
                    result = model.transcribe(file, language=language, fp16=True)
                    res = result["text"]
                else:
                    model = whisper.load_model(model_size, device="cpu")
                    result = model.transcribe(file, language=language, fp16=False)
                    res = result["text"]

                return res
            except Exception as err:
                print("an error occured while transcribing: ", err)
        elif model_type == "custom":
            model_folder = os.path.dirname(custom_model_path)
            model_folder = model_folder + "/"
            print("model file: ", custom_model_path)
            print("model fodler: ", model_folder)
            try:
                if torch.cuda.is_available():
                    model = whisper.load_model(custom_model_path, download_root=model_folder, device="cuda")
                    result = model.transcribe(file, language=language, fp16=True)
                    res = result["text"]
                else:
                    model = whisper.load_model(custom_model_path, download_root=model_folder, device="cpu")
                    result = model.transcribe(file, language=language, fp16=False)
                    res = result["text"]

                return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "huggingface":
            try:
                if torch.cuda.is_available():
                    pipe = pipeline("automatic-speech-recognition", model=hf_model_path, device="cuda")
                    result = pipe(file)
                    res = result['text']
                else:
                    pipe = pipeline("automatic-speech-recognition", model=hf_model_path, device="cpu")
                    result = pipe(file)
                    res = result['text']
                return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "assemblyAI":
            try:
                # Replace with your API key
                aai.settings.api_key = aai_api_key

                # You can set additional parameters for the transcription
                config = aai.TranscriptionConfig(
                    speech_model=aai.SpeechModel.nano,
                    language_code=language
                )

                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(file)

                if transcript.status == aai.TranscriptStatus.error:
                    print(transcript.error)
                    raise Exception(f"an error occured while transcribing: {transcript.error}")
                else:
                    res = transcript.text
                    return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "nemo":  # NeMo transcription logic
            try:
                # Load pre-trained ASR model from NeMo
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path='checkpoint.nemo')
                model.freeze() # inference mode
                model = model.to(device) # transfer model to device


                # Transcribe audio
                res = model.transcribe([file], batch_size=1,logprobs=False, language_id='ta')[0]  # Returns a list of transcriptions
                return res
            except Exception as err:
                raise Exception(f"an error occurred while transcribing with NeMo: {err}")
        else:
            raise Exception(f"model_type {model_type} is not supported")
    else:
        raise Exception("only 'base', 'tiny', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3' models are available.")

"""