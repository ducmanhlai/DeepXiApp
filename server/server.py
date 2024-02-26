import base64
import json
from flask import Flask,render_template,request, jsonify, send_file
import numpy as np
from scipy.io import wavfile
from io import BytesIO
import os
import soundfile as sf
import io
import librosa
from werkzeug.datastructures import FileStorage
from predict import predict
from deepxi.se_batch import Batch
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
            file.save(os.path.join('./upload/', file.filename))
            audio, sr = librosa.load(os.path.join('./upload/', file.filename))
            signal_mono= librosa.to_mono(audio)
            signal_resampled = librosa.resample(signal_mono,orig_sr = sr,target_sr= 16000)
            output_path = os.path.join('./upload/', file.filename.split(".")[0] + ".wav")
            sf.write(output_path, signal_resampled, 16000, format= 'wav')
            os.remove('./upload/'+file.filename)
            test_x, test_x_len, _, test_x_base_names = Batch('./upload')
            prediction = predict(test_x,test_x_len,test_x_base_names)
            os.remove('./upload/'+file.filename.split(".")[0] + ".wav")
            # wav = np.squeeze(prediction)
            # if isinstance(wav[0], np.float32):
            #     wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
            # audio_data = base64.b64encode(wav).decode('UTF-8')
            # return jsonify({"audio_data": audio_data})
            # data = {"snd": audio_data}
            # res = app.response_class(response=json.dumps(data),
            #         status=200,
            #         mimetype='audio/wav')
            # return res
            return jsonify({'source': '/static/predicts/mhanet-1.1c/e200/y/mmse-lsa/'+file.filename.split(".")[0] + ".wav"})
        except Exception as e:
            return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(port=8080,debug=True)
class Test:
    def hi():
        return 1