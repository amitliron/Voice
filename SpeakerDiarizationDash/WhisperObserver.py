from typing import Tuple, Union

import numpy as np
from pyannote.core import Annotation
from rx.core import Observer
import soundfile         as sf

class WhisperObserver(Observer):
    def __init__(self):
        super().__init__()
        self.i = 0

    def _extract_annotation(self, value: Union[Tuple, Annotation]) -> Annotation:
        if isinstance(value, tuple):
            return value[0]
        if isinstance(value, Annotation):
            return value
        msg = f"Expected tuple or Annotation, but got {type(value)}"
        raise ValueError(msg)

    def on_next(self, value: Union[Tuple, Annotation]):
        print("on_next")
        tmp = value[1].data
        speech = np.zeros(len(tmp))
        for i in range(len(tmp)):
            speech[i] = tmp[i][0]

        labels = value[1].labels
        if labels is None:
            labels = "None"
        duration = value[1].sliding_window.duration
        self.i = self.i + 1
        #sf.write(f"/home/amitli/Downloads/diart_tests/{self.i}_speaker_{labels}.wav", speech, 16000)
        print(f"{self.i}, len ={len(speech)}, Labels: {labels}, Duration: {duration}, Min: {min(speech)} Max: {max(speech)}")


    def on_error(self, error: Exception):
        #self.patch()
        print("error")

    def on_completed(self):
        print("complate")