from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from scipy.io import wavfile
from io import BytesIO
import os
import soundfile as sf
from pydub import AudioSegment
import librosa
from werkzeug.datastructures import FileStorage
import glob
from predict import predict
from deepxi.se_batch import Batch
app = Flask(__name__)


def delete(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)


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
            Path("./upload").mkdir(parents=True, exist_ok=True)
            file.save(os.path.join('./upload/', file.filename))
            audio, sr = librosa.load(os.path.join('./upload/', file.filename))
            signal_mono = librosa.to_mono(audio)
            signal_resampled = librosa.resample(
                signal_mono, orig_sr=sr, target_sr=16000)
            duration = librosa.get_duration(y=signal_resampled, sr=16000)
            if duration <= 23:
                output_path = os.path.join(
                    './upload/', file.filename.split(".")[0] + '16000' + ".wav")
                sf.write(output_path, signal_resampled, 16000, format='wav')
                os.remove('./upload/'+file.filename)
                test_x, test_x_len, _, test_x_base_names = Batch('./upload')
                predict(test_x, test_x_len, test_x_base_names)
                delete('./upload')
                return jsonify({'source': '/static/predicts/mhanet-1.1c/e200/y/mmse-lsa/'+file.filename.split(".")[0]+'16000' + ".wav"})
            else:
                os.remove('./upload/'+file.filename)
                name = file.filename.split(".")[0] + '16000'
                segment_length_ms = 23*16000
                count = 1
                for start_time in range(0, len(signal_resampled), segment_length_ms):
                    end_time = min(start_time + segment_length_ms,
                                   len(signal_resampled))
                    audio_segment = signal_resampled[start_time:end_time]
                    Path("./upload/"+name).mkdir(parents=True, exist_ok=True)
                    output_path = os.path.join(
                        './upload/', name + '/' + str(count) + '.wav')
                    sf.write(output_path, audio_segment, 16000, format='wav')
                    count += 1
                test_x, test_x_len, _, test_x_base_names = Batch(
                    './upload/'+name)
                predict(test_x, test_x_len, test_x_base_names, name)

                delete('./upload/'+name)
                os.rmdir('./upload/'+name)
                # predicted_audio = np.array([])
                # for filename in range(1,len(os.listdir('./static/predicts/'+name+'/mhanet-1.1c/e200/y/mmse-lsa'))+1):
                #             file_path = os.path.join('./static/predicts/'+name+'/mhanet-1.1c/e200/y/mmse-lsa', str(filename)+ '.wav')
                #             a, s = librosa.load(file_path)
                #             predicted_audio = np.concatenate((predicted_audio,a))
                # output_path = './static/predicts/mhanet-1.1c/e200/y/mmse-lsa/'+file.filename.split(".")[0]+'16000' + ".wav"
                # wav = np.squeeze(predicted_audio)
                # if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
                # sf.write(output_path, wav, s,format= 'wav')
                combines_sounds = AudioSegment.from_wav(
                    './static/predicts/'+name+'/mhanet-1.1c/e200/y/mmse-lsa/' + str(1) + '.wav')
                for filename in range(2, count):
                    file_path = os.path.join(
                        './static/predicts/'+name+'/mhanet-1.1c/e200/y/mmse-lsa', str(filename) + '.wav')
                    a = AudioSegment.from_wav(file_path)
                    combines_sounds = combines_sounds+a
                Path("./static/predicts/mhanet-1.1c/e200/y/mmse-lsa/").mkdir(parents=True, exist_ok=True)
                combines_sounds.export('./static/predicts/mhanet-1.1c/e200/y/mmse-lsa/' +
                                       file.filename.split(".")[0]+'16000' + ".wav", format="wav")
                return jsonify({'source': '/static/predicts/mhanet-1.1c/e200/y/mmse-lsa/'+file.filename.split(".")[0]+'16000' + ".wav"})
        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)


class Test:
    def hi():
        return 1
