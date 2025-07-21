import os
import json
import librosa
import torch
import laion_clap
import argparse
import torch.nn.functional as F


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--generation_result_path', type=str, default="")
    args.add_argument('--target_json_path', type=str, default="")
    return args.parse_args()


def main():
    args = get_args()
    
    generation_result_path = args.generation_result_path
    target_json_path = args.target_json_path
    
    device = torch.device("cuda")
    
    model = laion_clap.CLAP_Module(enable_fusion=False).to(device)
    model.load_ckpt('/mnt/vepfs/audio/share/jiangyuxuan/weights/weights/clap/630k-audioset-best.pt')
    
    score_list = []
    
    for file in os.listdir(generation_result_path):
        
        if not file.endswith(".wav"):
            continue

        wave_path = os.path.join(generation_result_path, file)
        json_path = os.path.join(target_json_path, file.replace(".wav", ".json"))
        
        with open(json_path, "r") as r_f:
            json_data = json.load(r_f)
        
        prompt = json_data["prompt"]
        
        audio_data, _ = librosa.load(wave_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
    
        audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
        audio_embed = torch.from_numpy(audio_embed)
        
        text_embed = model.get_text_embedding(prompt, use_tensor=True)
        
        score = F.cosine_similarity(audio_embed.cpu(), text_embed.cpu()).item()
        score_list.append(score)
    
    print()
    print(generation_result_path)
    print(generation_result_path)
    print(sum(score_list) / len(score_list))
    print()


if __name__ == '__main__':
    main()
