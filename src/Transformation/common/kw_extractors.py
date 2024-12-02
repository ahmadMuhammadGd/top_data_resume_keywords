from abc import ABC, abstractmethod

# keywords extractor part
class IKeywordExtractor(ABC):
    @abstractmethod
    def exec(self, text:str)->list[tuple[str, float]]:
        raise NotImplementedError

    
class KeywordExtractor:
    def __init__(self, keyword_extractor_object: IKeywordExtractor):
        self.keo = keyword_extractor_object
    
    def run(self, text:str):
        return self.keo.exec(text)

# implement your extractors here or in seperated files in theis directory
import yake
class YAKEKeywordExtractor(IKeywordExtractor):
    def exec(self, text:str):
        kw_extractor = yake.KeywordExtractor()
        return kw_extractor.extract_keywords(text)
