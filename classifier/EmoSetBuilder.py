from datasets import DatasetBuilder, DatasetInfo, GeneratorBasedBuilder
from datasets import SplitGenerator, Features, ClassLabel, Value, Image
import os, csv, json
from datasets import load_dataset


def load_emo_set(data_dir):
    emo_set_builder = EmoSetBuilder(dataset_path=data_dir)
    emo_set_builder.download_and_prepare()
    dataset = emo_set_builder.as_dataset()
    return dataset


class EmoSetBuilder(GeneratorBasedBuilder):

    def __init__(self, dataset_path):
        with open(os.path.join(dataset_path, "info.json")) as file:
            info = json.load(file)
        self.label_names = [None] * len(info["label2idx"])
        for label, idx in info["label2idx"].items():
            self.label_names[idx] = label
        self.dataset_path = dataset_path
        super().__init__()

    def _info(self):
        return DatasetInfo(
            description="EmoSet",
            features=Features({
                "img": Image(),
                "label": ClassLabel(names=self.label_names)
            }),
        )

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name="train", gen_kwargs={"file_path": os.path.join(self.dataset_path, "train.json")}
            ), SplitGenerator(
                name="val", gen_kwargs={"file_path": os.path.join(self.dataset_path, "val.json")}
            ), SplitGenerator(
                name="test", gen_kwargs={"file_path": os.path.join(self.dataset_path, "test.json")}
            ),
        ]

    def _generate_examples(self, file_path):
        with open(file_path) as file:
            data = json.load(file)  
        for idx, (label_str, image_path, annotation_path) in enumerate(data):
            image_path = os.path.join(self.dataset_path, image_path)
            annotation_path = os.path.join(self.dataset_path, annotation_path)
            with open(annotation_path) as file:
                row = json.load(file)
            yield idx, {
                "img": image_path,
                "label": label_str,
            }

if __name__ == "__main__":
    data_dir = "/work/pi_juanzhai_umass_edu/gehaozhang/EmoSet/data"
    dataset = load_emo_set(data_dir)
    print(dataset)
    dataset.save_to_disk("/work/pi_juanzhai_umass_edu/gehaozhang/EmoSet-118K-hf")