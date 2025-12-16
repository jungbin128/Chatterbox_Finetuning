class T3Config(dict):
    """
    T3 semantic model configuration.
    dict를 상속해서 cfg["key"] 접근이 가능하도록 설계.
    """

    def __init__(self):
        super().__init__({
            # model size
            "dim": 512,
            "heads": 8,
            "layers": 6,
            "dropout": 0.1,

            # vocabulary sizes
            "text_vocab": 30000,     # UTF-8 byte 기반 토큰 가정
            "speech_vocab": 2048,    # semantic token vocab

            # max lengths
            "max_text": 512,
            "max_speech": 1024,

            # special tokens
            "bos": 1,
            "eos": 2,
            "pad": 0,
        })

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
