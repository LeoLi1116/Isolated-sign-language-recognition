from flask import Flask, request, jsonify, render_template
import os
import uuid
import cv2
import numpy as np
from model import get_mediapipe, real_time_asl, choose_output
import mediapipe as mp
# import tflite_runtime.interpreter as tflite
    
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['VIDEO'] = 'ASL_videos'
app.config['MODELS'] = 'tflite_models'

GRU_model = tf.lite.Interpreter(os.path.join(app.config['MODELS'], "GRU_model.tflite"))
transformer_model = tf.lite.Interpreter(os.path.join(app.config['MODELS'], "transformer100.tflite"))
LSTM_model = tf.lite.Interpreter(os.path.join(app.config['MODELS'], "LSTM_model.tflite"))
ensemble_model = tf.lite.Interpreter(os.path.join(app.config['MODELS'], "ensemble_model.tflite"))


model_map = {
    'GRU': GRU_model,
    'LSTM': LSTM_model,
    'Transformer': transformer_model,
    'Ensemble': ensemble_model
}


@app.route('/')
def index():
    return render_template('index.html', video='bye', processed_video_path='', prediction_results='')


@app.route('/process', methods=['POST'])
def process_video():
    # if 'video' not in request.files:
    #     return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.form['video']
    # filename = f"{uuid.uuid4()}.mp4"
    # original_video_path = os.path.join(app.config['VIDEO'], filename)
    # video_file.save(original_video_path)
    
    model_name = request.form['model']
    
    video_path = 'static/videos/' + video_file + '.mp4'
    
    # Process the video using the provided function
    processed_video_path = get_mediapipe(video_path)
    print(processed_video_path)

    # Load the processed video and predict using the provided ML model
    sign, top5, top10 = real_time_asl(video_path, model_map[model_name], model_name)
    sign, top5, top10 = choose_output(model_name, video_file, sign, top5, top10)
    
    prediction = f"Prediction: {sign}\n Top 5: {top5}\n Top 10: {top10}"

    # Return the processed video URL and prediction result
    # processed_video_url = f'/static/uploads/{os.path.basename(processed_video_path)}'

    return render_template('index.html', video=video_file, processed_video_path=processed_video_path, prediction_results=prediction)

if __name__ == 'main':
    app.run(debug=True)
