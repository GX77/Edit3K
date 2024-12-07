import torch
import json
from PIL import Image
import os
import torch.nn.functional as F
from data.datasets.until_function import _transform
from tqdm import tqdm
import argparse
from mm_video.modeling.model import MODEL_REGISTRY
from decord import VideoReader
from decord import cpu
import numpy as np

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_frames(video_path, max_frames=16, normalize=None):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_step = total_frames // max_frames
        frame_indices = np.arange(0, total_frames, frame_step)[:max_frames]
        frames = vr.get_batch(frame_indices).asnumpy()
        frame_list = [normalize(Image.fromarray(frame)) for frame in frames]
        frames = torch.stack(frame_list)
    except:
        print(video_path)
        frames = torch.zeros(max_frames, 3, 224, 224)
    return frames


def load_model(model_path, device):
    checkpoint = torch.load(model_path)
    model = MODEL_REGISTRY.get(checkpoint["config"].MODEL.NAME)(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    guide_query = checkpoint['guide_query'].to(device)
    return model, guide_query

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='output/edit_1014/checkpoint_1.pth', type=str)
    parser.add_argument("--videos_dir", default="raw_videos_iclr_all/", type=str)
    parser.add_argument("--save_path", default="edit_embedding.json")
    parser.add_argument("--all_edit", default="edit_3k.json")
    parser.add_argument("--resolution", default=(224, 224), type=tuple)
    parser.add_argument("--max_frames", default=16, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model, guide_query = load_model(args.model_path, device)
    transform = _transform(args.resolution)
    edit_id2material_id_list = load_json(args.all_edit)["test"]
    embedding_list = []
    embedding_id_list = []
    for edit_id, material_id_list in tqdm(edit_id2material_id_list.items()):
        temp = []
        for material_id in material_id_list[:3]:
            video_path = args.videos_dir + material_id
            if not os.path.exists(video_path):
                print(video_path + "NOT FOUND!")
                continue
            video_frames = load_frames(video_path, normalize=transform, max_frames=args.max_frames).to(device)
            with torch.no_grad():
                video_embedding = model.generate_embedding(video_frames.unsqueeze(0), guide_query)
            temp.append(video_embedding)
        if len(temp) == 0:
            continue
        embedding_id_list.append(edit_id)
        embedding_list.append(torch.mean(torch.cat(temp, dim=0), dim=0))

    embedding_list = F.normalize(torch.stack(embedding_list), p=2, dim=1)
    data = {}
    data["id_list"] = embedding_id_list
    data["embedding_list"] = embedding_list.tolist()
    save_json(data, args.save_path)