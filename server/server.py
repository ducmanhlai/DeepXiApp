from flask import Flask,render_template,request, jsonify
import numpy as np
from scipy.io import wavfile
from io import BytesIO
import librosa
from predict import predict
app = Flask(__name__)

@app.route('/')
def hello_world():
    """Returns a simple 'Hello, world!' message."""
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Convert MP3 file to numpy array
            mp3_bytes = file.read()
            wav_bytes = BytesIO(mp3_bytes)
            sr, data = wavfile.read(wav_bytes)
             # Chuyển đổi sang mono
            signal_mono = librosa.to_mono(data)
    # Chuyển đổi sample rate
            signal_resampled = librosa.resample(signal_mono,orig_sr = sr,target_sr= 16000)
            prediction= predict(signal_resampled,1,file.filename)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(port=8080,debug=True)
class Test:
    def hi():
        return 1