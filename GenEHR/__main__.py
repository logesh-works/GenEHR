from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from GenEHR.genehr import Transcriptor
from .convert_to_wav import convert_to_wav

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploaded_files"
LOG_FOLDER = "logs"
VOICES_FOLDER = "voices"
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}
LANGUAGE = "ta"
MODEL_SIZE = "large-v2"
QUANTIZATION = True
ACCESS_TOKEN = "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"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Renders the HTML template for uploading files."""
    return render_template('index.html')

@app.route('/doctor', methods=['GET'])
def doctor_page():
    """Renders the doctor.html page."""
    return render_template('doctor.html')

@app.route('/upload_doctor_audio', methods=['POST'])
def upload_doctor_audio():
    """Handles uploading and storing doctor's audio."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    doctor_name = request.form.get('doctor_name', 'Unknown').strip()

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Secure doctor name for directory creation
        secure_dr_name = secure_filename(doctor_name)
        doctor_folder = os.path.join(VOICES_FOLDER, f"dr_{secure_dr_name}")
        os.makedirs(doctor_folder, exist_ok=True)

        # Save the original file temporarily
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(doctor_folder, filename)
        file.save(temp_file_path)

        # Convert the file to WAV format if it's an MP3
        if filename.lower().endswith('.mp3'):
            wav_file_path = convert_to_wav(temp_file_path)
            # Delete the temporary MP3 file
            os.remove(temp_file_path)
            filename = os.path.basename(wav_file_path)  # Update filename to WAV version

        # Save the WAV file
        final_file_path = os.path.join(doctor_folder, filename)
        os.rename(wav_file_path, final_file_path)

        return jsonify({"status": "success", "message": f"Audio saved in {final_file_path}"})

    return jsonify({"error": "File type not allowed"}), 400
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    patient_name = request.form.get('patient_name', 'Unknown')
    patient_age = request.form.get('patient_age', 'Unknown')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process the file using Transcriptor
            transcriptor = Transcriptor(
                patient_name=patient_name,
                patient_age=patient_age,
                dialog_audio=filename,
                file="uploaded_files/recording.mp3",
                log_folder=LOG_FOLDER,
                language=LANGUAGE,
                modelSize=MODEL_SIZE,
                ACCESS_TOKEN=ACCESS_TOKEN,
                voices_folder=VOICES_FOLDER,
                quantization=QUANTIZATION
            )
            result = transcriptor.nemo()

            return jsonify({"status": "success", "result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

