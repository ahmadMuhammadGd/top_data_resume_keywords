from transformers import pipeline
from abc import ABC, abstractmethod
class ITextClassefier(ABC):
    @abstractmethod
    def exec(self, text:str, candidate_labels:list[str])->dict:
        """
        labels MUST be sorted by scores and the output form MUST look like this:
        {
            'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 
            'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 
            'scores': [0.504, 0.479, 0.013, 0.003, 0.002]
        }
        """
        raise NotImplementedError

class BartLargeMnli(ITextClassefier):
    def __init__(self):
        self.pipe = pipeline(model="facebook/bart-large-mnli")
        
    def exec(self, text:str, candidate_labels:list[str])->dict:
        return self.pipe(text, candidate_labels=candidate_labels)

class TextClassefier:
    def __init__(self, classefier:ITextClassefier):
        self.classefier = classefier
    def run(self, text:str, candidate_labels:list[str]):
        return self.classefier.exec(text=text, candidate_labels=candidate_labels)