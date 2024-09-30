from GenEHR.genehr import Transcriptor

file = "obama_zach.wav" 
voices_folder = "voices" 
language = "ta"         
log_folder = "logs"    
modelSize = "large-v2"    
quantization = False   
ACCESS_TOKEN = "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" 


transcriptor = Transcriptor(file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder, quantization)

print("Running Nemo")
res = transcriptor.nemo()


""" print("running whisper....")
res = transcriptor.whisper()


print("running faster-whisper....")
res = transcriptor.faster_whisper() """