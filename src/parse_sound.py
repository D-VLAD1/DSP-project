import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data" / "voice-icar-federico-ii-database-1.0.0"

_GAIN_RE = re.compile(
    r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\((?P<baseline>[+-]?\d+)\))?"
    r"(?:/(?P<units>\S+))?$"
)
_INFO_RE = re.compile(
    r"#<age>:\s*(?P<age>\d+)\s*"
    r"<sex>:\s*(?P<sex>\S+)\s*"
    r"<diagnoses>:\s*(?P<diagnoses>.*?)\s*"
    r"<medications>:\s*(?P<medications>.*)\s*$"
)
_SAMPLE_DTYPES = {16: np.int16, 32: np.int32}


@dataclass(frozen=True)
class SignalHeader:
    filename: str
    sample_format: int
    gain: float
    baseline: int
    units: str
    adc_resolution: int
    adc_zero: int
    initial_value: int
    checksum: int
    block_size: int
    description: str


@dataclass(frozen=True)
class HeaHeader:
    record_name: str
    n_signals: int
    sampling_frequency: float
    n_samples: int
    signals: tuple[SignalHeader, ...]
    age: int | None = None
    sex: str | None = None
    diagnoses: str | None = None
    medications: str | None = None


@dataclass(frozen=True)
class Recording:
    header: HeaHeader
    info: dict
    samples: np.ndarray


def parse_hea(path: str | Path) -> HeaHeader:
    lines = Path(path).read_text().splitlines()
    records = [line for line in lines if line and not line.startswith("#")]
    if len(records) < 2:
        raise ValueError(f"{path} has no signal specification")

    name, n_signals_text, frequency_text, n_samples_text, *_rest = records[0].split()
    n_signals = int(n_signals_text)
    signal_lines = records[1:]
    if len(signal_lines) != n_signals:
        raise ValueError(
            f"{path} declares {n_signals} signals but has {len(signal_lines)}"
        )

    info = _parse_info(next((line for line in lines if line.startswith("#")), ""))
    return HeaHeader(
        record_name=name,
        n_signals=n_signals,
        sampling_frequency=float(frequency_text),
        n_samples=int(n_samples_text),
        signals=tuple(_parse_signal(line) for line in signal_lines),
        **info,
    )


def load_samples(header: HeaHeader, directory: str | Path) -> np.ndarray:
    if header.n_signals != 1:
        raise ValueError(f"{header.record_name} has {header.n_signals} signals")

    signal = header.signals[0]
    dtype = _SAMPLE_DTYPES.get(signal.sample_format)
    if dtype is None:
        raise ValueError(f"unsupported sample format {signal.sample_format}")

    digital = np.fromfile(Path(directory) / signal.filename, dtype=dtype)
    if digital.size != header.n_samples:
        raise ValueError(
            f"{signal.filename} has {digital.size} samples, header says {header.n_samples}"
        )
    if signal.gain == 0:
        return digital.astype(np.float64)
    return (digital.astype(np.float64) - signal.baseline) / signal.gain


def parse_voice_info(path: str | Path) -> dict:
    """Parse a voiceNNN-info.txt questionnaire into one row of columns."""
    fields: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        fields[key] = value.strip()

    def text(key: str) -> str:
        return fields[key]

    def optional(key: str) -> str | None:
        value = fields[key]
        if value == "" or value.upper() == "NU":
            return None
        return value

    def whole(key: str) -> int | None:
        value = optional(key)
        return None if value is None else int(value)

    return {
        "record_id": text("ID"),
        "age": int(text("Age:")),
        "gender": text("Gender:"),
        "diagnosis": text("Diagnosis:"),
        "occupation": optional("Occupation status:"),
        "vhi_score": int(text("Voice Handicap Index (VHI) Score:")),
        "rsi_score": int(text("Reflux Symptom Index (RSI) Score:")),
        "smoker": text("Smoker:"),
        "cigarettes_per_day": whole("Number of cigarettes smoked per day:"),
        "alcohol_consumption": text("Alcohol consumption:"),
        "alcoholic_drinks_per_day": optional(
            "Number of glasses containing alcoholic beverage drinked in a day"
        ),
        "water_litres_per_day": float(
            text("Amount of water's litres drink every day:").replace(",", ".")
        ),
        "carbonated_beverages": text("Carbonated beverages:"),
        "carbonated_drinks_per_day": optional("Amount of glasses drinked in a day"),
        "tomatoes": text("Tomatoes:"),
        "coffee": text("Coffee:"),
        "coffee_cups_per_day": whole("Number of cups of coffee drinked in a day"),
        "chocolate": text("Chocolate:"),
        "chocolate_grams_per_day": optional("Gramme of chocolate eaten in  a day"),
        "soft_cheese": text("Soft cheese:"),
        "soft_cheese_grams_per_day": optional("Gramme of soft cheese eaten in a day"),
        "citrus_fruits": text("Citrus fruits:"),
        "citrus_fruits_per_day": optional("Number of citrus fruits eaten in a day"),
    }


def parse(sound_name: str) -> Recording:
    header = parse_hea(DATA_FOLDER / f"{sound_name}.hea")
    info = parse_voice_info(DATA_FOLDER / f"{sound_name}-info.txt")
    return Recording(header, info, load_samples(header, DATA_FOLDER))


def play_sound(sound_name: str) -> None:
    recording = parse(sound_name)
    sd.play(
        recording.samples.astype(np.float32),
        samplerate=recording.header.sampling_frequency,
    )
    sd.wait()


def _parse_signal(line: str) -> SignalHeader:
    filename, sample_format, gain_field, resolution, adc_zero, initial, checksum, block_size, *description = line.split()
    gain, baseline, units = _parse_gain(gain_field, int(adc_zero))
    return SignalHeader(
        filename=filename,
        sample_format=int(sample_format),
        gain=gain,
        baseline=baseline,
        units=units,
        adc_resolution=int(resolution),
        adc_zero=int(adc_zero),
        initial_value=int(initial),
        checksum=int(checksum),
        block_size=int(block_size),
        description=" ".join(description),
    )


def _parse_gain(field: str, adc_zero: int) -> tuple[float, int, str]:
    match = _GAIN_RE.fullmatch(field)
    if match is None:
        raise ValueError(f"invalid gain field {field!r}")
    baseline = match.group("baseline")
    return (
        float(match.group("gain")),
        int(baseline) if baseline is not None else adc_zero,
        match.group("units") or "mV",
    )


def _parse_info(line: str) -> dict[str, int | str | None]:
    empty = {"age": None, "sex": None, "diagnoses": None, "medications": None}
    if not line:
        return empty
    match = _INFO_RE.fullmatch(line)
    if match is None:
        return empty
    age, sex, diagnoses, medications = match.groups()
    return {
        "age": int(age),
        "sex": sex,
        "diagnoses": diagnoses,
        "medications": medications,
    }
