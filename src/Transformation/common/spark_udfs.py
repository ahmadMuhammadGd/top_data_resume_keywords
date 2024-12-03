
from src.Transformation.common.kw_extractors import KeywordExtractor
from src.Transformation.common.text_clustering import ITextClassifier, TextClassifier
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType
from pyspark.sql.functions import udf

# spark udfs
def create_keyword_extractor_udf(kw_extractor_bc):
    @udf(returnType=ArrayType(
        StructType([
            StructField("keyword", StringType(), False),
            StructField("weight", FloatType(), False)
        ])
    ))
    def udf_extract_keywords(text):
        if not text:
            return None
        kw_extractor = kw_extractor_bc.value
        results = KeywordExtractor(kw_extractor).run(text)
        return [(str(kw), float(weight)) for kw, weight in results]
    
    return udf_extract_keywords



def create_text_classefier_udf(txt_classefier_bc, candidate_lables_bc):
    @udf
    def text_classefier(text):
        if not text:
            return None
        kw_extractor = txt_classefier_bc.value
        candidate_lables = candidate_lables_bc.value
        results = TextClassifier(kw_extractor).run(text, candidate_lables)
        
        if not results:
            return None
        
        if results['scores'][0] > 0.2:
            return str(results['labels'][0])
        
        else:
            return None
    
    return text_classefier

def create_clean_country_udf(iso2_list_bc, iso3_list_bc, country_names_list_bc):
    iso2_list                       =   iso2_list_bc.value
    iso3_list                       =   iso3_list_bc.value
    country_names_list              =   country_names_list_bc.value
    
    standarized_iso2_list           = [x.strip().lower() for x in iso2_list]
    standarized_iso3_list           = [x.strip().lower() for x in iso3_list]
    standarized_country_names_list  = [x.strip().lower() for x in country_names_list]
    
    standardized_countries = list(zip(standarized_iso2_list, standarized_iso3_list, standarized_country_names_list))

    @udf
    def clean_country_udf(entry: str):
        if not entry:  
            return None
        
        cleaned_entry = entry.strip().lower().split(' ')
        
        if 'anywhere' in cleaned_entry and len(cleaned_entry)==1:
            return 'Anywhere'
        
        for iso2, iso3, country_name in standardized_countries:
            if (iso2 in cleaned_entry) or (iso3 in cleaned_entry) or all([x in cleaned_entry for x in country_name.split(' ')]):
                return country_name.capitalize()
        
        return None  

    return clean_country_udf