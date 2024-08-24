module load cuda/10.0 cudnn/7.5_cuda-10.0
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
wget "https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/files/?p=%2Fresnet50_weights_best_acc.tar&dl=1" -O resnet50_weights_best_acc.tar