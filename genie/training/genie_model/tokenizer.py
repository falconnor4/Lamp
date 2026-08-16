from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|mask|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
    "<|image|>",
    "<|audio|>",
    "<|cursor|>",
    "<|state|>",
    "<|action|>",
]

PAD_ID, MASK_ID, BOS_ID, EOS_ID, SEP_ID, IMAGE_ID, AUDIO_ID, CURSOR_ID, STATE_ID, ACTION_ID = range(len(SPECIAL_TOKENS))


def train_bpe(files, vocab_size, out_path):
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tokenizer.train(files, trainer)
    tokenizer.post_processor = processors.ByteLevel()
    tokenizer.save(out_path)
    return tokenizer


def load(path):
    return Tokenizer.from_file(path)
