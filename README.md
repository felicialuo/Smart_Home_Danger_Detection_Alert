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
- For Video Finetuned CLIP, go to `./ViFi-CLIP` and create a new folder named `ckpts`. Download this [checkpoint](https://mbzuaiac-my.sharepoint.com/personal/uzair_khattak_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fuzair%5Fkhattak%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fvifi%5Fclip%5Fweights%2Fzero%5Fshot%5Fweights%2Fvifi%5Fclip%5F10%5Fepochs%5Fk400%5Ffull%5Ffinetuned%2Epth&parent=%2Fpersonal%2Fuzair%5Fkhattak%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fvifi%5Fclip%5Fweights%2Fzero%5Fshot%5Fweights&ga=1) and move it to `./ViFi-CLIP/ckpts`.


## Run Inferences
- To run CLAP inference, open [clap_inference.ipynb](./clap_inference.ipynb). Specify label csv file path and video path in the first cell and run all.
- To run Video Finetuned CLIP inference, open [ViFi-CLIP_inference.ipynb](./ViFi-CLIP/ViFi-CLIP_inference.ipynb). Specify label csv file path and video path in the first cell and run all.