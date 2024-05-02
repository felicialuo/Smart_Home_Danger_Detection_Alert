from msclap import CLAP
import torch.nn.functional as F
import numpy as np
import torch
import os
from moviepy.editor import VideoFileClip
import sounddevice as sd
import numpy as np
from torch.nn import functional as F
from scipy.io.wavfile import write
import time
import threading
import queue
from IPython.display import display, clear_output
import csv
import joblib
import socket

def get_labels(csv_path='datasets/home_labels.csv'):
    label2id = {}
    id2label = {}
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)

        for i, row in enumerate(csv_reader):
            class_name = row[0]
            label2id[class_name] = i
            id2label[i] = class_name
    class_labels = list(label2id.keys())
    return label2id, id2label, class_labels

def get_predictions(audio_filename, window_size = 2, sample_rate = 16000):
    label2id, id2label, class_labels = get_labels()

    print("Loading model...")
    # Load model (Choose between versions '2022' or '2023')
    # The model weight will be downloaded automatically if `model_fp` is not specified
    clap_model = CLAP(version='2023', use_cuda=False)

    print("Extracting text embeddings...")
    # Extract text embeddings
    text_embeddings = clap_model.get_text_embeddings([f"This is a sound of {c}" for c in class_labels])

    print("Listening...")

    while True:   
        LOCAL_IP = '0.0.0.0'  # Listen on all network interfaces
        PORT = 50007
        # REMOTE_IP = '192.168.1.154'
        REMOTE_IP = "172.26.128.166"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((bind_ip, bind_port))
            server_socket.listen(1)
            print("Server is listening for connections...")
    
            while True:
                # Accept a client connection
                conn, addr = server_socket.accept()
                print(f"Connection attempted from {addr[0]}")
    
                # Check if the connection is from the allowed IP
                # if addr[0] == allowed_ip:
                    # print(f"Connected by {addr}")
                with conn:
                    with open('received_output.wav', 'wb') as f:
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            f.write(data)
    
                    print("File received successfully.")
    
                    response_message = "Thank you, file received!"
    
                    # Send a confirmation message back to the client
                    conn.sendall(response_message.encode('utf-8'))

        # Extract audio embeddings
        audio_embeddings = clap_model.get_audio_embeddings([audio_filename])
        # Compute similarity between audio and text embeddings
        similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

        similarity = F.softmax(similarities, dim=1)
        values, indices = similarity[0].topk(5)

        detected = False
        clap_results = []
        print("\nCLAP predictions:")
        for value, index in zip(values, indices):
            index = index.item()
            value = round(value.item() * 100, 4)
            clap_results.append(index)
            clap_results.append(value)
            print(id2label[index], value)
            if id2label[index] in ["Crying", "Gunshot", "Glass breaking"] and value > 50: 
                print(f"ALERT: {id2label[index]} detected!")
                detected = True
                break
            
        # Ensemble
        if not detected:
            vclip_results = [-1, 0] * 5
            trained_ensemble = joblib.load('trained_RF_ensemble.joblib')
            X_test = np.expand_dims(np.hstack([vclip_results, clap_results]), axis=0)

            y_pred = trained_ensemble.predict(X_test)
            print("\nEmsemble prediction:", id2label[y_pred[0]])
            if id2label[y_pred[0]] in ["Crying", "Gunshot", "Glass breaking"]: 
                print(f"ALERT: {id2label[y_pred[0]]} detected!")


        os.remove(audio_filename) 
        # if detected:
        #     filename = f"recording_{int(time.time())}.wav"
        #     os.rename(temp_filename, filename)
        #     print(f"Alert! Glass breaking detected in {filename}")
        # else:
        #     os.remove(temp_filename) 



if __name__ == "__main__":
    
    get_predictions(audio_filename, window_size = 2, sample_rate = 16000)

    
