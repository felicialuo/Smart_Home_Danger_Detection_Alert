import joblib
import csv
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from ViFi_CLIP.utils.config import get_config
from ViFi_CLIP.utils.logger import create_logger
from ViFi_CLIP.utils.config import get_config
from ViFi_CLIP.trainers import vificlip
from ViFi_CLIP.datasets.pipeline import *
import torch.nn.functional as F
from msclap import CLAP
from moviepy.editor import VideoFileClip
import os


def read_results(csv_path):
    results = []
    labels = []
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            file_name, label, top1_pred, top1_conf, top2_pred, top2_conf, top3_pred, top3_conf, top4_pred, top4_conf, top5_pred, top5_conf = row
            labels.append(label)
            results.append([top1_pred, top1_conf, top2_pred, top2_conf, top3_pred, top3_conf, top4_pred, top4_conf, top5_pred, top5_conf])

    results = np.asarray(results).astype(float)
    labels  = np.asarray(labels).astype(int)
    return results, labels

def get_labels(csv_path='datasets/home_labels.csv'):
    label2id = {}
    id2label = {}
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)

        for i, row in enumerate(csv_reader):
            class_name = row[0]
            label2id[class_name] = i
            id2label[i] = class_name

    return label2id, id2label

def prepare_vclip(class_labels):
    config = 'ViFi_CLIP/configs/zero_shot/train/k400/16_16_vifi_clip.yaml'
    output_folder_name = "ViFi_CLIP/outputs"
    pretrained_model_path = "ViFi_CLIP/ckpts/vifi_clip_10_epochs_k400_full_finetuned.pth"

    # Step 1:
    # Configuration class 
    class parse_option():
        def __init__(self):
            self.config = config
            self.output =  output_folder_name   # Name of output folder to store logs and save weights
            self.resume = pretrained_model_path
            # No need to change below args.
            self.only_test = True
            self.opts = None
            self.batch_size = None
            self.pretrained = None
            self.accumulation_steps = None
            self.local_rank = 0
    args = parse_option()
    config = get_config(args)
    # # logger
    # logger = create_logger(output_dir=args.output, name=f"{config.MODEL.ARCH}")
    # logger.info(f"working dir: {config.OUTPUT}")

    # Step 2:
    # Create the ViFi-CLIP models and load pretrained weights
    model = vificlip.returnCLIP(config,
                                # logger=logger,
                                class_names=class_labels,)
    model = model.float()

    # logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    load_state_dict = checkpoint['model']
    # now remove the unwanted keys:
    if "module.prompt_learner.token_prefix" in load_state_dict:
        del load_state_dict["module.prompt_learner.token_prefix"]

    if "module.prompt_learner.token_suffix" in load_state_dict:
        del load_state_dict["module.prompt_learner.token_suffix"]

    if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
        del load_state_dict["module.prompt_learner.complete_text_embeddings"]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in load_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    # load params
    msg = model.load_state_dict(new_state_dict, strict=False)
    # logger.info(f"resume model: {msg}")

    # Step 3: 
    # Preprocessing for video
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    pipeline = Compose(val_pipeline)

    return model, pipeline

def inference_vclip(file_path, model, pipeline):
    if file_path.endswith(".mp4") or file_path.endswith(".avi"):
        video_path = file_path
    else: 
        print(f"This file {file_path} has no video!")
        return [-1, 0] * 5

    dict_file = {'filename': video_path, 'tar': False, 'modality': 'RGB', 'start_index': 0, 'gt_label': video_path.split('/')[-2]}
    video = pipeline(dict_file)
    video_tensor = video['imgs'].unsqueeze(0).float()

    # Inference through ViFi-CLIP
    with torch.no_grad():
        similarities = model(video_tensor)

    similarity = F.softmax(similarities, dim=1)
    values, indices = similarity[0].topk(5)

    # output top5 class_idx and conf
    out = []
    for value, index in zip(values, indices):
        out.append(index.item())
        out.append(round(value.item() * 100, 4))

    return out

def inference_clap(file_path, class_labels):
    if file_path.endswith(".mp4"):
        video = VideoFileClip(file_path)
        if video.audio is None:
            print(f"This video {file_path} has no audio!")
            return [-1, 0] * 5
        else:
            audio_path = 'datasets/temp_audio.wav'
            audio_dir = "/".join(audio_path.split("/")[:-1])
            # print(audio_dir)
            if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)
            audio_arr = video.audio.write_audiofile(audio_path, verbose=False)
            # print(f"Audio saved to {audio_path}")
    else: audio_path = file_path

    with torch.no_grad():
        # Load model (Choose between versions '2022' or '2023')
        # The model weight will be downloaded automatically if `model_fp` is not specified
        clap_model = CLAP(version = '2023', use_cuda=False)

        # Extract text embeddings
        text_embeddings = clap_model.get_text_embeddings([f"This is a sound of {c}"for c in class_labels])

        # Extract audio embeddings
        audio_embeddings = clap_model.get_audio_embeddings([audio_path])

        # Compute similarity between audio and text embeddings 
        similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

    similarity = F.softmax(similarities, dim=1)
    values, indices = similarity[0].topk(5)
    
    # output top5 class_idx and conf
    out = []
    for value, index in zip(values, indices):
        out.append(index.item())
        out.append(round(value.item() * 100, 4))

    return out





if __name__ == "__main__":
    # Input video
    file_path = 'datasets/test/v_BabyCrawling_g10_c04.avi'

    # Class label csv path
    labels_csv_path = 'datasets/home_labels.csv'

    # get labels
    label2id, id2label = get_labels(csv_path=labels_csv_path)

    # # prepare vclip
    # vclip_model, vclip_pipeline = prepare_vclip(label2id.keys())

    
    import time
    start_time = time.time()

    # print("---VCLIP")
    # vclip_results = inference_vclip(file_path, vclip_model, vclip_pipeline)
    # for i in [0, 2, 4, 6, 8]:
    #     if vclip_results[i] == -1: continue
    #     if i % 2 == 0: print(id2label[vclip_results[i]], vclip_results[i+1])

    print("---CLAP")
    clap_results = inference_clap(file_path, label2id.keys())
    for i in [0, 2, 4, 6, 8]:
        if clap_results[i] == -1: continue
        if i % 2 == 0: print(id2label[clap_results[i]], clap_results[i+1])
    



    # Ensemble
    vclip_results = [-1, 0] * 5
    trained_ensemble = joblib.load('trained_RF_ensemble.joblib')

    X_test = np.expand_dims(np.hstack([vclip_results, clap_results]), axis=0)

    y_pred = trained_ensemble.predict(X_test)
    print("Prediction:", id2label[y_pred[0]])


        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference time is {elapsed_time} seconds.")