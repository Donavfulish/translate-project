import os

class PhoMTDataset:
    def __init__(self, src_tgt_lang="en-vi"):
        if src_tgt_lang != "en-vi":
            raise NotImplementedError(
                f"Bộ dữ liệu cho cặp ngôn ngữ '{src_tgt_lang}' chưa được hỗ trợ."
            )

        self.src_lang, self.tgt_lang = src_tgt_lang.split("-")
        self.prefix_path = os.path.join("data", "input", "raw")

        self.splits = {}
        for split in ("train", "test", "dev"):
            self.splits[split] = self._load_split(split)

    def _load_split(self, split):
        src_file = os.path.join(self.prefix_path, split, f"{split}.{self.src_lang}")
        tgt_file = os.path.join(self.prefix_path, split, f"{split}.{self.tgt_lang}")

        with open(src_file, "r", encoding="utf-8") as f_src:
            src = [line.strip() for line in f_src]

        with open(tgt_file, "r", encoding="utf-8") as f_tgt:
            tgt = [line.strip() for line in f_tgt]

        return {"source": src, "target": tgt}

    def train(self): return self.splits["train"]
    def test(self): return self.splits["test"]
    def dev(self): return self.splits["dev"]