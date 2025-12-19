import os


class PhoMTDataset:
    def __init__(self, src_tgt_lang: str = "en-vi"):
        if src_tgt_lang != "en-vi":
            raise NotImplementedError(
                f"Bộ dữ liệu cho cặp ngôn ngữ '{src_tgt_lang}' chưa được hỗ trợ."
            )

        self.src_lang, self.tgt_lang = src_tgt_lang.split("-")

        # LẤY PATH TUYỆT ĐỐI DỰA TRÊN FILE NÀY
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # app/services/datasets → app/data/input/raw
        self.prefix_path = os.path.abspath(
            os.path.join(
                base_dir,
                "..", "..", "..",   # ra tới app/
                "data",
                "input",
                "raw",
            )
        )

        if not os.path.exists(self.prefix_path):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu: {self.prefix_path}"
            )

        self.splits = {}
        for split in ("train", "test", "dev"):
            self.splits[split] = self._load_split(split)

    def _load_split(self, split: str):
        src_file = os.path.join(
            self.prefix_path,
            split,
            f"{split}.{self.src_lang}",
        )
        tgt_file = os.path.join(
            self.prefix_path,
            split,
            f"{split}.{self.tgt_lang}",
        )

        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Không tìm thấy file: {src_file}")
        if not os.path.exists(tgt_file):
            raise FileNotFoundError(f"Không tìm thấy file: {tgt_file}")

        with open(src_file, "r", encoding="utf-8") as f_src:
            src = [line.strip() for line in f_src]

        with open(tgt_file, "r", encoding="utf-8") as f_tgt:
            tgt = [line.strip() for line in f_tgt]

        if len(src) != len(tgt):
            raise ValueError(
                f"Số dòng không khớp ({split}): src={len(src)}, tgt={len(tgt)}"
            )

        return {"source": src, "target": tgt}

    def train(self):
        return self.splits["train"]

    def test(self):
        return self.splits["test"]

    def dev(self):
        return self.splits["dev"]
