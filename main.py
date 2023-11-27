from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pickle
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F


# Đặt thiết bị là GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained("DrishtiSharma/whisper-large-v2-vietnamese")
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


def pad_or_trim(array, length: int = 480000 , *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """

    if type(array) == list:
        array = np.array(array)

    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

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
    batch_size = 256
    for i in tqdm(range(0, len(data), batch_size)):
        batch_records = data[i:i + batch_size]
        input_texts = [pad_or_trim(record['array']) for record in batch_records]

        input_features = processor(input_texts, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True).input_features.to(device)
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        decoded_texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        for j, record in enumerate(batch_records):
            record['api_sentence'] = decoded_texts[j]
            record['compare'] = compare_script(record['sentence'], decoded_texts[j])

    return data

def verify_from_file_pkl(pkl_file_path):
    data = read_file_pkl(pkl_file_path)
    data = verify_dataset(data)
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    verify_from_file_pkl("./Data54.pkl")
    # data = read_file_pkl("./Data54.pkl")
    # print(data[:4])
