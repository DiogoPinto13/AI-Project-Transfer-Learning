##
## First step
Instal libraries using:  pip install -r requirements.txt

## Optional 
If cuda is installed and has a different version, then disable it to avoid errors in code by using: export CUDA_VISIBLE_DEVICES=""

## Second step
To train the model on the original dataset can be utilized the original link on kaggle: https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-97-cnn , the file can also be found in here: "**traffic-signs-image-classification-97-5b89f9.ipynb**"

## Third step
To run the code to train with transfer learning and also knowledge distillation (and also do the 30 experiments) it's just necessary to run:
python3 train.py

## Additional steps
If train.py is runned it will output the results of the experiments and that information can be fed to **results.py** to calculate the confidence intervals and then in **plotResults.py** for the plots.
