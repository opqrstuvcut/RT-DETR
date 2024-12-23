"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import glob
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images: list[Image.Image], read_paths, labels, boxes, scores, thrh=0.6):
    for i, (im, p) in enumerate(zip(images, read_paths)):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for b, l in zip(box, lab):
            draw.rectangle(list(b), outline="red", width=3)
            draw.text(
                (b[0], b[1]),
                text=str(l),
                fill="blue",
            )

        im.save(f"./inferences/{Path(p).name}.jpg")


def create_mini_batch(iterable, n):
    iter_steps = len(iterable) // n + (len(iterable) % n > 0)
    for i in range(iter_steps):
        yield iterable[i * n : (i + 1) * n]


def main(
    args,
):
    """main"""

    output_dir = Path("inferences")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    if args.im_dir:
        image_files = glob.glob(args.im_dir + "/*")
    else:
        image_files = [args.im_file]

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    for batch in create_mini_batch(image_files, n=8):
        images = []
        np_images = []
        orig_sizes = []
        read_paths = []
        for p in batch:
            try:
                im_pil = Image.open(p).convert("RGB")
            except Exception:
                continue

            read_paths.append(p)

            w, h = im_pil.size
            orig_sizes.append([w, h])

            np_images.append(transforms(im_pil))
            images.append(im_pil)

        output = model(
            torch.stack(np_images).to(args.device),
            torch.tensor(orig_sizes).to(args.device),
        )
        labels, boxes, scores = output

        draw(images, read_paths, labels, boxes, scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--im-file",
        type=str,
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument(
        "--im-dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
