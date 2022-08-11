from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForest, RandomForestModel
from time import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

from pyspark.streaming import StreamingContext
# from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer, KafkaConsumer

import pandas as pd
from pandas import json_normalize
import json

# create Spark Context with the specified configuration
def initSparkContext():
    spark = SparkSession.builder.appName('Diabetes Data').getOrCreate()

    return spark


def loadDataset(dataSet):
    # convert data frame to Pandas
    dataSet.toPandas()

    # How many rows we have
    # dataFrame.count()

    # The names of our columns
    # dataFrame.columns

    # Basics stats from our columns
    # dataFrame.describe().toPandas()

    # convert columns type

    dataSet = dataSet.select(
        col('PatientID').cast('int'),
        col('Pregnancies').cast('int'),
        col('PlasmaGlucose').cast('int'),
        col('DiastolicBloodPressure').cast('int'),
        col('TricepsThickness').cast('int'),
        col('SerumInsulin').cast('int'),
        col('BMI').cast('float'),
        col('DiabetesPedigree').cast('float'),
        col('Age').cast('int'),
        col('Diabetic').cast('int')
    )
    dataSet.toPandas()

    # Assemble all the features with VectorAssembler
    requiredFeatures = [
        'PatientID',
        'Pregnancies',
        'PlasmaGlucose',
        'DiastolicBloodPressure',
        'TricepsThickness',
        'SerumInsulin',
        'BMI',
        'DiabetesPedigree',
        'Age'
    ]

    assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')

    # add Vector features in data set
    transformedData = assembler.transform(dataSet)

    transformedData.toPandas()

    return transformedData


def trainingModel(trainingData):
    startTime = time()

    model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=3,
                                         maxDepth=8)
    endTime = time()

    print("\nTime to train model: %.3f seconds\n" % (endTime - startTime))

    # print("\n\nnumber Trees ================= : ", model.numTrees())

    # print("\n\ntotal number Trees ================= : ", model.totalNumNodes())

    # print("\n\nModel ================= : ", model)

    # print("\nModel to Debug String ================= : ")
    # print(model.toDebugString())

    return model


def predictionsModel(model, data):
    startTime = time()

    predictResult = model.predict(data.map(lambda x: x.features))

    endTime = time()

    print("\nTime to predict model: %.3f seconds\n" % (endTime - startTime))

    return predictResult

def calculationAccuracy(predictionsResult, testDataLabels):
    # Convert result Model from RDD Type to DataFrame Type and named predictionsDF
    predictionsDF = predictionsResult.map(lambda x: (x,)).toDF(["predictions"])
    predictionsDF = list(predictionsDF.select('predictions').toPandas()['predictions'])

    # Convert result from RDD Type to DataFrame Type and named labelsDF
    labelsDF = testDataLabels.map(lambda x: (x,)).toDF(["labels"])
    labelsDF = list(labelsDF.select('labels').toPandas()['labels'])

    # Get random 10 rows and count labelsDF
    # print("list labels DataFrame : ", labelsDF)
    # print("count labels DataFrame : ", len(labelsDF))

    labelsAndPredictions = zip(labelsDF, predictionsDF)

    # print("list labelsAndPredictions : ", list(labelsAndPredictions))
    # print("type labelsAndPredictions : ", type(labelsAndPredictions))
    # print("count labelsAndPredictions : ", len(list(labelsAndPredictions)))

    filterLabelsAndPredictions = filter(lambda x: x[0] != x[1], list(labelsAndPredictions))

    filterLabelsAndPredictionsCount = len(list(filterLabelsAndPredictions))

    testDataCount = float(len(labelsDF))

    testErr = filterLabelsAndPredictionsCount / testDataCount * 100

    return testErr


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spark = initSparkContext()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")


    # load Dataset in data Frame type
    dataFrame = spark.read.csv("diabetes.csv", header=True)

    Dataset = loadDataset(dataFrame)

    # convert data set from `Data Frame` type to `RDD` type
    rddObj = Dataset.rdd

    # Split Data Set to training data and test data
    (training_data, test_data) = rddObj.randomSplit([0.7, 0.3])

    new_training_data = training_data.map(lambda row: LabeledPoint(row["Diabetic"], Vectors.dense(row['features'])))
    new_test_data = test_data.map(lambda row: LabeledPoint(row["Diabetic"], Vectors.dense(row['features'])))

    # ------------------------------------------------------------------------------------------------------

    model = trainingModel(new_training_data)

    # Evaluate model on test instances and compute test error
    predictions = predictionsModel(model, new_test_data)

    # Split regional result from test data and named labels
    labels = new_test_data.map(lambda lp: lp.label)

    Accuracy = calculationAccuracy(predictions, labels)

    print('\n=========================================\n||\t\t\t\t\t||')

    print('|| Test Accuracy = ', Accuracy, "||")

    print('||\t\t\t\t\t||\n=========================================\n')

    # dataKafka = spark \
    #     .readStream \
    #     .format("kafka") \
    #     .option("kafka.bootstrap.servers", "localhost:9092") \
    #     .option("subscribe", "input_recommend_product") \
    #     .load()
    #
    # dF = dataKafka.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    #
    # query = dataKafka.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
    #     .writeStream \
    #     .format("console") \
    #     .start()
    #
    # query.awaitTermination()
    #
    # print("\n\nData Kafka => ", dataKafka)
    #
    # print("\n\nds => ", dF)

    # rawQuery = dsraw \
    #     .writeStream \
    #     .queryName("qraw") \
    #     .format("memory")\
    #     .start()

    # print("raw Query => ", rawQuery)

    # alertQuery = ds \
    #     .writeStream \
    #     .queryName("qalerts") \
    #     .format("memory") \
    #     .start()
    #
    # print("alert Query => ", alertQuery)
    #
    # raw = spark.sql("select * from qraw")
    # raw.show()

    # batch duration, here i process for each second
    # ssc = StreamingContext(sc, 5)

    consumer = KafkaConsumer('input_recommend_product',
         bootstrap_servers='localhost:9092',
         value_deserializer=lambda m: json.loads(m.decode('utf8'))
    )

    json_list = []

    for msg in consumer:
        json_list.append(msg.value)
        jsonToDataFrame = spark.read.json(sc.parallelize(json_list))

        rowNewRDD = loadDataset(jsonToDataFrame).rdd

        # print("count row New RDD 1 : ",  rowNewRDD.count())
        # print("value row New RDD 1 : ",  rowNewRDD.collect())

        rowNewRDD = rowNewRDD.map(lambda row: LabeledPoint(row["Diabetic"], Vectors.dense(row['features'])))

        # print("count row New RDD 2 : ",  rowNewRDD.count())
        # print("value row New RDD 2 : ",  rowNewRDD.collect())

        # Evaluate model on test instances and compute test error
        predictionsNewRow = predictionsModel(model, rowNewRDD)

        result = predictionsNewRow.collect()[0]

        if(result == 0):
            print("The patient is  : Injured")
        else:
            print("The patient is  : Healthy")

    # ---------------------------------------------------------------------------------
"""

# batch duration, here i process for each second
ssc = StreamingContext(sc, 1)

kafkaStream = KafkaUtils.createStream(ssc, '127.0.0.1:2181', 'test-consumer-group', {'input_event': 1})

lines = kafkaStream.map(lambda x: process_events(x))

lines.foreachRDD(push_back_to_kafka)

ssc.start()
ssc.awaitTermination()

"""

# bin/kafka-console-producer.sh --topic input_recommend_product --bootstrap-server localhost:9092
# bin/kafka-console-consumer.sh --topic input_recommend_product --from-beginning --bootstrap-server localhost:9092

#  /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 PySpark_Kafka.py


# {"PatientID": 158000, "Pregnancies": 0, "PlasmaGlucose": 171, "DiastolicBloodPressure": 80, "TricepsThickness": 34, "SerumInsulin": 23, "BMI": 43.50972593, "DiabetesPedigree": 1.213191354, "Age": 21, "Diabetic": 2 }
