# Smart Home Danger Detection + Alert
Final-project-in-progress for 17-422/722,05-499/899, Spring 2024: Building User-Focused Sensing Systems.  
This code is built upon official repositories of [CLAP](https://github.com/microsoft/CLAP) and [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP).
 
## Environment Setup
This environment has been tested on MacBook Pro with M1 Pro chip.
```
conda env create -f environment.yaml
```


## Download Checkpoints
- For CLAP, weights will be downloaded automatically.
- For Video Finetuned CLIP, go to `./ViFi-CLIP` and create a new folder named `ckpts`. Download this [checkpoint](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EW0shb6XYDxFi3BH6DT70rgBPDwgW_knQ8jDsarxINXezw?e=RbixXc) and move it to `./ViFi-CLIP/ckpts`.


## Run Inferences
- To run CLAP inference, open [clap_inference.ipynb](./clap_inference.ipynb). Specify label csv file path and video path in the first cell and run all.
- To run Video Finetuned CLIP inference, open [ViFi-CLIP_inference.ipynb](./ViFi-CLIP/ViFi-CLIP_inference.ipynb). Specify label csv file path and video path in the first cell and run all.

## Run Pi Code
- In the smarthome-pi folder, run python3 main.py

## Run Webapp
= In the webapp folder, run 
```
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver
```
