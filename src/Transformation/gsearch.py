from src.Transformation.common.kw_extractors import YAKEKeywordExtractor
from src.Transformation.common.text_clustering import BartLargeMnli
from src.Transformation.common.spark_udfs import create_keyword_extractor_udf, create_clean_country_udf, create_text_classefier_udf
from global_variables import LANDING_DIR, LUKES_DATASET_NAME, CLEAN_LOCATION_DATASET_NAME

import os
gsearch_file_path = os.path.join(LANDING_DIR, LUKES_DATASET_NAME)
countries_file_path = os.path.join(LANDING_DIR, CLEAN_LOCATION_DATASET_NAME)

txt_classefier = BartLargeMnli()
kw_extractor = YAKEKeywordExtractor()

import pandas as pd
countries_df = pd.read_csv(countries_file_path)


iso2 = countries_df["iso2"].dropna().astype(str).tolist()
iso3 = countries_df["iso3"].dropna().astype(str).tolist()
country_names = countries_df["country"].dropna().astype(str).tolist()

del countries_df

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, col, explode

spark = SparkSession.builder.appName("kw_extractor").getOrCreate()


try:
    kw_extractor_bc         = spark.sparkContext.broadcast(kw_extractor)
    txt_classefier_bc       = spark.sparkContext.broadcast(txt_classefier)
    iso2_bc                 = spark.sparkContext.broadcast(iso2)
    iso3_bc                 = spark.sparkContext.broadcast(iso3)
    country_names_bc        = spark.sparkContext.broadcast(country_names)
    candidate_lables_bc     = spark.sparkContext.broadcast(["Data Engineer", "Data Analyst", "Business Analyst", "BI Developer", "Data Scientist"])
    
    keyword_extractor_udf   = create_keyword_extractor_udf(kw_extractor_bc)    
    text_classefier_udf     = create_text_classefier_udf(txt_classefier_bc, candidate_lables_bc)
    clean_country_udf       = create_clean_country_udf(iso2_list_bc=iso2_bc, iso3_list_bc=iso3_bc, country_names_list_bc=country_names_bc)

    df = spark.read.option("header",True).csv(gsearch_file_path)
    
    enriched = df.select(
                df.title
                , df.location
                , df.description
            ).limit(50)\
            .filter(
                df.title.isNotNull()
                & df.location.isNotNull()
                & df.description.isNotNull()
            ).withColumn(
                "country", clean_country_udf(df.location)
            ).withColumn(
                "kw_arr", keyword_extractor_udf(df.description)
            ).filter(
                col("Country").isNotNull()
            ).withColumn(
                "cleaned_job_title", text_classefier_udf(df.title)
            )\
            .groupBy(
                col("cleaned_job_title"), 
                col("Country")
            ).agg(
                collect_list(col("kw_arr")).alias("keywords") 
            ).withColumn(
                "keyword", explode(col("keywords"))
            ).drop("keywords")  
            
    enriched.show()
    # reduced.show()
except:
    raise

finally:
    spark.stop()
    pass