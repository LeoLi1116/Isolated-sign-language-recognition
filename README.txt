Code submission for CSC249 final project
Junfei Liu  jliu137@u.rochester.edu
Chunhao Li  cli79@u.rochester.edu

The code folder will include the code or link to the public code we used for this projects. There are some notebooks 
that are modified based on public notebooks. In this case, we will note reference at the beginning. Most of the
notebooks are editted and run on kaggle with NVIDIA Tesla P100.

This folder will include the following:

1. data-processing-transformer-training.ipynb: 
    It includes data preprocessing and basic transformer architecture with embeddings of lips, hands, and faces and 
    positional embedding. Its reference is noted at the beginning and codes copied from public sources are also noted 
    as annotations. The current saved version is the test trail for transformer with no positional embeddings.

2. gru.ipynb:
    It includes the GRU based method.

3. lstm.ipynb:
    It includes the LSTM based method.

x. ASL-web-demo:
    It includes the web demo shown during the presentation. It is developed using Python Flask. Run instructions are:
    enter the /flask directory and follow standard start server procedure.


The methods tested with public notebooks are:
1. Ensemble of ensembles (on the shoulders ensemble): https://www.kaggle.com/code/aikhmelnytskyy/gislr-tf-on-the-shoulders-ensamble-v2-0-69
2. CNN+3trans speedup (champion of the competition): https://www.kaggle.com/code/kolyaforrat/cnn-3trans-speedup
3. KerasTuner: https://www.kaggle.com/code/aynoji/gislr-tf-kerastuner
4. ViT without separate embedding: https://www.kaggle.com/code/dhk13491/gislr-simple-vit-pyorch-to-tflite-baseline