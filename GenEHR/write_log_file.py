import os
from datetime import datetime
from .summarize import summarize

def write_log_file(patient_name,patient_age,dialog_audio,common_segments, log_folder, file_name, language):

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    #---------------------log file part-------------------------
        
    current_time = datetime.now().strftime('%H%M%S')

    file_name = os.path.splitext(os.path.basename(file_name))[0]

    log_file = log_folder + "/" + f"{patient_name}({patient_age})" + "_" + current_time + "_" + language + ".txt"
    
    lf=open(log_file,"wb")

    entry = f"""Patient Name: {patient_name}
Patient Age: {patient_age}
Date and time : {datetime.now()}
    
conversation: 
    
    """
    texts = ""
    for segment in common_segments:
        start = segment[0]
        end = segment[1]
        text = segment[2]
        speaker = segment[3]
        
        if text != "" and text != None:
            entry += f"{speaker} ({start} : {end}) : {text}\n"
            texts += f"{speaker} : {text[0]} ,"
    
    
    entry += f"""
Summarized Points:
    
{summarize(texts)['response']}
    """
    lf.write(bytes(entry.encode('utf-8')))      
    lf.close()

    return entry

    # -------------------------log file end-------------------------
