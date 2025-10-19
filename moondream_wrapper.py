import time
from PIL import Image
import spacy
import moondream as md

class MoondreamExtractor:
    """
    Minimal wrapper over moondream vl model - loads model and provides:
      - get_attributes(image, prompt) -> raw text
      - parse_attributes(text) -> list of attribute tokens (colors, nouns, adj_noun combos)
    """
    def __init__(self, model_path="./moondream-0_5b-int8.mf"):
        print("Loading Moondream VL model...")
        self.vl = md.vl(model=model_path)
        print("Moondream ok")
        self.nlp = spacy.load("en_core_web_sm")
        print("spaCy loaded")

    def get_attributes(self, image: Image.Image, prompt: str) -> str:
        # encode & query; measure time for debug
        img_enc = self.vl.encode_image(image)
        t0 = time.time()
        resp = self.vl.query(img_enc, prompt)
        t1 = time.time()
        # resp is expected to be a dict containing 'answer'
        txt = resp.get("answer", "")
        # print(f"Moondream query took {t1-t0:.2f}s")
        return txt

    def parse_attributes(self, text: str):
        """
        Extract nouns and adjective+noun pairs using spaCy.
        Returns a list of tokens like: ['white_shirt', 'backpack', 'glasses']
        """
        doc = self.nlp(text)
        out = set()
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN"):
                out.add(tok.text.lower())
            if tok.pos_ == "ADJ":
                head = tok.head
                if head.pos_ in ("NOUN", "PROPN"):
                    out.add(f"{tok.text.lower()}_{head.text.lower()}")
        return list(out)

    @staticmethod
    def simple_extract(text: str):
        keywords = ["robot", "hood", "head", "platform"]
        colors = ["white", "blue", "green", "red", "black", "yellow"]
        found = [k for k in keywords if k in text.lower()]
        found_colors = [f"{c}_medium" for c in colors if c in text.lower()]
        return {"attributes": found, "colors": found_colors}
