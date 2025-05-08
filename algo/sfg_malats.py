# algo/sfg_malats.py
from .fg_malats import FGMALATS

class SFGMALATS(FGMALATS):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("fg_mode",   "smooth")
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return "SFGMALATS"
