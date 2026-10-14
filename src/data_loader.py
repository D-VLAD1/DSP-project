from parse_sound import DATA_FOLDER, parse

from sklearn.model_selection import train_test_split
import numpy as np


def load_voices() -> tuple:
    paths = sorted(DATA_FOLDER.glob("voice*.dat"))
    samples, labels = [], []

    for path in paths:
        parsed = parse(path.stem)
        samples.append(parsed.samples)

        label = parsed.info["diagnosis"]
        labels.append(label != "healthy")

    min_length = min(len(sample) for sample in samples)
    return np.array([sample[:min_length] for sample in samples]), np.array(labels)


def get_data(test_size: float = 0.2, random_state: int = 42) -> tuple:
    voices, labels = load_voices()
    X_train, X_test, y_train, y_test = train_test_split(voices, labels, test_size=test_size,
                                                        random_state=random_state, stratify=labels)

    return (X_train, y_train), (X_test, y_test)

