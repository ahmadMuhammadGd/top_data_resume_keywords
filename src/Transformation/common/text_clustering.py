from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class ITextClassifier(ABC):
    @abstractmethod
    def exec(self, text: str, candidate_labels: List[str]) -> Dict:
        pass

class BartMnli(ITextClassifier):
    def __init__(self):
        self._pipe = None
    
    @property
    def pipe(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(model="facebook/bart-large-mnli")
        return self._pipe
    
    def exec(self, text: str, candidate_labels: List[str]) -> Dict:
        return self.pipe(text, candidate_labels=candidate_labels)

from typing import List, Dict


class SSTuningALBERT:
    def __init__(self):
        self._pipe = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _initialize_model(self):
        if self._tokenizer is None or self._model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            global torch
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")
            self._model = AutoModelForSequenceClassification.from_pretrained("DAMO-NLP-SG/zero-shot-classify-SSTuning-ALBERT")
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.to(self._device).eval()

    def _build_analyzer(self, text: str, candidate_labels: List[str]) -> Dict:
        if not candidate_labels:
            raise ValueError("'candidate_labels' can't be None")

        self._initialize_model()

        labels = [f"{label}." if not label.endswith('.') else label for label in candidate_labels]
        options = ' '.join(f"({chr(65 + i)}) {label}" for i, label in enumerate(labels))
        input_text = f"{options} {self._tokenizer.sep_token} {text}"

        encoding = self._tokenizer([input_text], truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[:, :len(labels)]
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            predictions = sorted(zip(candidate_labels, probs.tolist()), key=lambda x: x[1], reverse=True)

        return {
            'sequence': text,
            'labels': [x[0] for x in predictions],
            'scores': [x[1] for x in predictions],
        }

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._build_analyzer
        return self._pipe

    def exec(self, text: str, candidate_labels: List[str]) -> Dict:
        return self.pipe(text, candidate_labels)

        
class LazyFuzzyClassifier(ITextClassifier):
    def __init__(self):
        self._pipe = None

    @staticmethod
    def __wratio_extractor(text: str, candidate_labels: List[str]) -> List[tuple]:
        from thefuzz import fuzz
        return [(label, fuzz.WRatio(text, label)) for label in candidate_labels]
    
    @staticmethod
    def __ratio_extractor(text: str, candidate_labels: List[str]) -> List[tuple]:
        from thefuzz import fuzz
        return [(label, fuzz.ratio(text, label)) for label in candidate_labels]
    
    @staticmethod
    def __partial_ratio_extractor(text: str, candidate_labels: List[str]) -> List[tuple]:
        from thefuzz import fuzz
        return [(label, fuzz.partial_ratio(text, label)) for label in candidate_labels]

    @staticmethod
    def __prcess_extractor(text: str, candidate_labels: List[str]) -> List[tuple]:
        from thefuzz import process
        return process.extract(text, candidate_labels, limit=len(candidate_labels))
    
    def __build_pipe(self):

        def pipe(text: str, candidate_labels: List[str]) -> Dict:
            extractors_list = [
                self.__prcess_extractor,
                self.__wratio_extractor,
                self.__ratio_extractor,
                self.__partial_ratio_extractor
            ]
            
            for extractor in extractors_list:
                res = extractor(text, candidate_labels)
                res.sort(key=lambda tup: tup[1], reverse=True)
                scores = [x[1] for x in res]
                pass_condition = len(set(scores)) == len(scores)
                if pass_condition:
                    return {
                        'sequence': text,
                        'labels': [x[0] for x in res],
                        'scores': [x[1] / 100 for x in res]
                    }
            
            return None

        self._pipe = pipe

    def pipe(self, text: str, candidate_labels: List[str]) -> Dict:

        if self._pipe is None:
            self.__build_pipe()
        return self._pipe(text, candidate_labels)

    def exec(self, text: str, candidate_labels: List[str]) -> Dict:
        return self.pipe(text, candidate_labels)

    
class TextClassifier:
    def __init__(self, classifier:ITextClassifier):
        self.classifier = classifier
    def run(self, text:str, candidate_labels:list[str]):
        return self.classifier.exec(text=text, candidate_labels=candidate_labels)
    
    
    
# classifier = LazyFuzzyClassifier()

# # Example text and candidate labels
# text = "Analyze data pipelines and optimize workflows."
# candidate_labels = ["Data Analyst", "Data Engineer", "Software Engineer"]

# # Executing the classifier
# result = classifier.exec(text, candidate_labels)

# # Outputting the result
# print(result)
