import os
from src.Transformation.common.kw_extractors import YAKEKeywordExtractor
from src.Transformation.common.text_clustering import LazyFuzzyClassifier
from src.Transformation.common.spark_udfs import create_keyword_extractor_udf, create_clean_country_udf, create_text_classefier_udf
from global_variables import LANDING_DIR, LUKES_DATASET_NAME, CLEAN_LOCATION_DATASET_NAME

gsearch_file_path = os.path.join(LANDING_DIR, LUKES_DATASET_NAME)
countries_file_path = os.path.join(LANDING_DIR, CLEAN_LOCATION_DATASET_NAME)

txt_classefier = LazyFuzzyClassifier()
kw_extractor = YAKEKeywordExtractor()

import pandas as pd
countries_df = pd.read_csv(countries_file_path)


iso2 = countries_df["iso2"].dropna().astype(str).tolist()
iso3 = countries_df["iso3"].dropna().astype(str).tolist()
country_names = countries_df["country"].dropna().astype(str).tolist()

del countries_df



from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, col, flatten, explode, desc


spark = SparkSession.builder.appName("kw_extractor").getOrCreate()


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



enriched = (
    df.select(col("title"), col("location"), col("description")).filter(
        col("location").isNotNull() & col("description").isNotNull()
    )
    .limit(100)
    .withColumn("country", clean_country_udf(col("location")))
    .withColumn("kw_arr", keyword_extractor_udf(col("description")))
    # .withColumn("clean_title", text_classefier_udf(col("title")))
    # .filter(col("country").isNotNull())
    # .drop("location")
    # .groupBy(col("clean_title"), col("country"))
    # .agg(collect_list(col("kw_arr")).alias("kw_arr"))
    # .withColumn("keywords", flatten(col("kw_arr")))
    # .drop("kw_arr")
)

enriched.show(truncate=False)