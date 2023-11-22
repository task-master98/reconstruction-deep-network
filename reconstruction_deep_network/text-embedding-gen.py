import os
from reconstruction_deep_network.trainer.trainer import ModelTrainer



def read_prompts_from_directory(directory):
    prompts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                prompt = file.read().strip()
                rotation_degree = filename.split('_')[-1].replace('.txt', '')
                prompts[rotation_degree] = prompt
    return prompts

def extract_common_prefix(filenames):
   
    prefix = os.path.commonprefix(filenames)
    return prefix.rstrip('_')

if __name__ == '__main__':
    trainer = ModelTrainer()
    device = 'cpu'

    mp3d_skybox_directory = "/Users/mario/Desktop/ECE 740/reconstruction-deep-network/reconstruction_deep_network/data/mp3d_skybox"
    embeddings_base_directory = "/Users/mario/Desktop/ECE 740/reconstruction-deep-network/reconstruction_deep_network/data/embeddings"

    for directory in os.listdir(mp3d_skybox_directory):
        if directory.startswith('.'): 
            continue

        blip3_directory = os.path.join(mp3d_skybox_directory, directory, "blip3")
        embeddings_directory = os.path.join(embeddings_base_directory, directory)

        if not os.path.exists(blip3_directory): 
            continue

        os.makedirs(embeddings_directory, exist_ok=True)

        text_files = sorted([f for f in os.listdir(blip3_directory) if f.endswith('.txt')])

        for i in range(0, len(text_files), 8):
            scene_files = text_files[i:i + 8]
            embeddings_dict = {}

            for file_name in scene_files:
                with open(os.path.join(blip3_directory, file_name), 'r') as file:
                    prompt = file.read().strip()
                    rotation_degree = file_name.split('_')[-1].replace('.txt', '')
                    x, y = trainer.encode_text(prompt, device)
                    print(f"Rotation: {rotation_degree}, Embedding shapes: {x.shape}, {y.shape}")
                    embeddings_dict[rotation_degree] = {'embeddings_1': x.cpu().numpy(), 'embeddings_2': y.cpu().numpy()}

            common_prefix = extract_common_prefix(scene_files)
            np.savez(os.path.join(embeddings_directory, f"{common_prefix}.npz"), **embeddings_dict)