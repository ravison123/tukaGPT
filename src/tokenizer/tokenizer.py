from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def train_tokenizer(file_path, output_path):
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(vocab_size=6000, min_frequency=2, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train([
        str(file_path)
    ], trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.save(str(output_path), pretty=True)


def encode(text, tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer.encode(text)

def decode(token_ids, tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer.decode(token_ids)

def split_data(file_path, train_data_path, valid_data_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    total_poems = data.split("<|endoftext|>")
    train_poems, valid_poems = train_test_split(total_poems, test_size=0.2, random_state=0)
    train_data = "<|endoftext|>".join(train_poems)
    train_data = train_data.strip()
    train_data = train_data + '\n' + "<|endoftext|>"
    valid_data = "<|endoftext|>".join(valid_poems)
    valid_data = valid_data.strip()
    valid_data = valid_data + '\n' + "<|endoftext|>"

    with open(train_data_path, 'w', encoding='utf-8') as f:
        f.write(train_data)

    with open(valid_data_path, 'w', encoding='utf-8') as f:
        f.write(valid_data)


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent
    data_file = ROOT / "data" / "tukaram_gatha_overall_data.txt"
    output_path = ROOT / "artifacts" / "bpe_tokenizer_huggingface.json"
    # train_tokenizer(file_path=data_file, output_path=output_path)
    #
    # example_text = 'गव्हाराचें ज्ञान अवघा रजोगुण । सुखवासी होऊन विषय भोगी ॥1॥'
    # print('Length of input example is: {}'.format(len(example_text)))
    # encoded_text = encode(example_text, output_path)
    #
    # decoded_text = decode(encoded_text.ids, output_path)
    #
    # assert decoded_text == example_text

    train_data_path = ROOT / "data" / "tukaram_gatha_train_data.txt"
    valid_data_path = ROOT / "data" / "tukaram_gatha_valid_data.txt"

    split_data(data_file, train_data_path, valid_data_path)

    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = f.read()

    train_data_encoded = encode(train_data, output_path)
    train_data_encoded = np.array(train_data_encoded.ids, dtype=np.int32)
    np.save(ROOT / "data" / "tukaram_gatha_train_encoded.npy", train_data_encoded)

    with open(valid_data_path, 'r', encoding='utf-8') as f:
        valid_data = f.read()

    valid_data_encoded = encode(valid_data, output_path)
    valid_data_encoded = np.array(valid_data_encoded.ids, dtype=np.int32)
    np.save(ROOT / "data" / "tukaram_gatha_valid_encoded.npy", valid_data_encoded)





