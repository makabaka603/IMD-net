import argparse
import torch
from models import IMDNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', default='imdnet.onnx')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    args = parser.parse_args()
    ckpt = torch.load(args.weights, map_location='cpu')
    cfg = ckpt['config']
    model = IMDNet(**cfg['model'])
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    x = torch.randn(1, 3, args.height, args.width)
    torch.onnx.export(model, x, args.output, opset_version=17, input_names=['input'], output_names=['restored'])


if __name__ == '__main__':
    main()
