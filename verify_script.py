from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pickle
from tqdm import tqdm
import torch
import h5py

# Đặt thiết bị là GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained("DrishtiSharma/whisper-large-v2-vietnamese")
print(processor)
model = WhisperForConditionalGeneration.from_pretrained("DrishtiSharma/whisper-large-v2-vietnamese")
model.to(device)  # Chuyển model sang GPU nếu có
forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

def remove_punt(str):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in str:
        if ele in punc:
            str = str.replace(ele, "")
    str = ' '.join(str.split())
    return str.lower()


def save_to_hdf5(file_path, data_list):
    with h5py.File(file_path, 'w') as hdf5_file:
        for i, item in enumerate(data_list):
            hdf5_file.create_dataset(f'data_{i}', data=item)
            

def compare_script(audio_script, whisper_scrpit):
    tmp_audio_script = audio_script
    tmp_whisper_script = whisper_scrpit
    
    tmp_audio_script = remove_punt(tmp_audio_script)
    tmp_whisper_script = remove_punt(tmp_whisper_script)
    if tmp_whisper_script == tmp_audio_script:
        return True
    return False

def read_file_pkl(pkl_file_path):
    with open(pkl_file_path,'rb') as f:
        data = pickle.load(f)
    return data

def verify_dataset(data):
    for record in tqdm(data):
        whisper_script = ""
        equals = False
        try:
            _ , array, script = record['path'], record['array'], record['sentence']
            input_features = processor(array, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True).input_features.to(device)
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            whisper_script = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            record['api_sentence'] = whisper_script
            equals = compare_script(script,whisper_script)
            record['compare'] = equals
        except:
            record['api_sentence'] = whisper_script
            record['compare'] = equals
    return data

def verify_from_file_pkl(pkl_file_path):
    data = read_file_pkl(pkl_file_path)
    data = verify_dataset(data)
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data, f)
    
if __name__ == "__main__":
    # verify_from_file_pkl("output_segment_audio/Data54.pkl")
    a = 5