from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from pathlib import Path

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


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent
    data_file = ROOT / "data" / "tukaram_gatha_overall_data.txt"
    output_path = ROOT / "artifacts" / "bpe_tokenizer_huggingface.json"
    train_tokenizer(file_path=data_file, output_path=output_path)

    example_text = 'गव्हाराचें ज्ञान अवघा रजोगुण । सुखवासी होऊन विषय भोगी ॥1॥'
    print('Length of input example is: {}'.format(len(example_text)))
    encoded_text = encode(example_text, output_path)

    decoded_text = decode(encoded_text.ids, output_path)

    assert decoded_text == example_text