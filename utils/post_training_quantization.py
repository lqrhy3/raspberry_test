import argparse
import torch
import os
import time
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
from models.detection.faced_model import FacedModelLite
from models.classification.mini_xception import MiniXception
from utils.utils import ImageToTensor


# POST TRAINING QUANTIZATION
def run_ptq():
    if TASK == 'detection':
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), ImageToTensor()])
    elif TASK == 'classification':
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.Grayscale(), ToTensor()])

    dataset = ImageFolder(PATH_TO_DATA, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    if TASK == 'detection':
        model = FacedModelLite(*MODEL_PARAMS)
    elif TASK == 'classification':
        model = MiniXception(['Anger', 'Happy', 'Neutral', 'Surprise'], *MODEL_PARAMS)

    load = torch.load(os.path.join(PATH_TO_MODEL, 'checkpoint.pt'), map_location='cpu')
    model.load_state_dict(load['model_state_dict'])
    model.to('cpu')
    model.eval()

    model.fuse_model()
    torch.backends.quantized.engine = ENGINE
    model.qconfig = torch.quantization.get_default_qconfig(ENGINE)
    torch.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, (image, _) in enumerate(dataloader, 1):
            image = torch.tensor(image, dtype=torch.float)
            print(f'Image #{i}')
            start = time.time()
            model(image)
            print('Time passed (s):', time.time() - start)
            print('===========')
            if i >= NUM_BATCHES:
                break

    torch.quantization.convert(model, inplace=True)
    torch.jit.save(torch.jit.trace(model, torch.ones_like(image)), os.path.join(PATH_TO_MODEL, 'model_quantized.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run post training quantization with calibration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', type=str, default='detection',
                        choices=['detection', 'classification'],
                        help='whether quantize detection model or classififcation')
    parser.add_argument('--path_to_model', type=str, default='log/detection/faced_model_lite', help='path to model')
    parser.add_argument('--path_to_data', type=str, default='data/detection/callibration_images',
                        help='path to calibration data')
    parser.add_argument('--model_params', type=str, default='9 2', help='mode initializing parameters (space separated)')
    parser.add_argument('--image_size', type=int, default=64, help='calibration image size')
    parser.add_argument('--batch_size', type=int, default=1, help='calibration batch size')
    parser.add_argument('--num_batches', type=int, default=5, help='number of calibration batches')
    parser.add_argument('--save_type', type=str, default='trace',
                        help='whether to use JIT.trace or to JIT.script to save quantized model',
                        choices=['trace', 'script'])
    opt = parser.parse_args()

    TASK = opt.task
    PATH_TO_MODEL = os.path.join('..', opt.path_to_model)
    PATH_TO_DATA = os.path.join('..', opt.path_to_data)
    MODEL_PARAMS = opt.model_params
    IMAGE_SIZE = opt.image_size
    BATCH_SIZE = opt.batch_size
    NUM_BATCHES = opt.num_batches
    SAVE_TYPE = opt.save_type
    ENGINE = 'qnnpack'

    run_ptq()
