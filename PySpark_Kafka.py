from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForest, RandomForestModel
from time import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    spark = SparkSession.builder.appName('Diabetes Data').getOrCreate()

    sc = spark.sparkContext

    df = spark.read.csv("diabetes.csv", header=True)

    df.toPandas()

    # How many rows we have
    df.count()

    # The names of our columns
    df.columns

    # Basics stats from our columns
    df.describe().toPandas()

    dataset = df.select(
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
    dataset.toPandas()

    # Assemble all the features with VectorAssembler
    required_features = [
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

    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    transformed_data = assembler.transform(dataset)

    transformed_data.toPandas()
    # ---------------------------------------------------------------------------------------------------- #

    rddObj = transformed_data.rdd

    (training_data, test_data) = rddObj.randomSplit([0.7, 0.3])

    new_training_data = training_data.map(lambda row: LabeledPoint(row["Diabetic"], Vectors.dense(row['features'])))
    new_test_data = test_data.map(lambda row: LabeledPoint(row["Diabetic"], Vectors.dense(row['features'])))

    # ------------------------------------------------------------------------------------------------------

    start_time = time()

    model = RandomForest.trainClassifier(new_training_data, numClasses=2, categoricalFeaturesInfo={}, numTrees=3, maxDepth=8)

    end_time = time()
    elapsed_time = end_time - start_time
    print("\n\nTime to train model: %.3f seconds" % elapsed_time)

    # print("\n\nnumber Trees ================= : ", model.numTrees())

    # print("\n\ntotal number Trees ================= : ", model.totalNumNodes())

    # print("\n\nModel ================= : ", model)


    print("\nModel to Debug String ================= : ")
    # print(model.toDebugString())

    # Evaluate model on test instances and compute test error
    predictions = model.predict(new_test_data.map(lambda x: x.features))

    # Convert result Model from RDD Type to DataFrame Type and named predictionsDF
    predictionsDF = predictions.map(lambda x: (x, )).toDF(["predictions"])
    predictionsDF = list(predictionsDF.select('predictions').toPandas()['predictions'])

    # Get random 10 rows and count predictionsDF
    # print('list predictions DataFrame : ', predictionsDF)
    # print("count predictions DataFrame : ", len(predictionsDF))

    # Split regional result from test data and named labels
    labels = new_test_data.map(lambda lp: lp.label)

    # Convert result from RDD Type to DataFrame Type and named labelsDF
    labelsDF = labels.map(lambda x: (x, )).toDF(["labels"])
    labelsDF = list(labelsDF.select('labels').toPandas()['labels'])

    # Get random 10 rows and count labelsDF
    # print("list labels DataFrame : ", labelsDF)
    # print("count labels DataFrame : ", len(labelsDF))

    labelsAndPredictions = zip(labelsDF, predictionsDF)

    # print("list labelsAndPredictions : ", list(labelsAndPredictions))
    print("type labelsAndPredictions : ", type(labelsAndPredictions))
    # print("count labelsAndPredictions : ", len(list(labelsAndPredictions)))

    # for l, n in labelsAndPredictions:
    #     print('labelsDF: ', l)
    #     print('predictionsDF: ', n)

    filterLabelsAndPredictions = filter(lambda x: x[0] != x[1], list(labelsAndPredictions))

    filterLabelsAndPredictionsCount = len(list(filterLabelsAndPredictions))

    print("count filterlabelsAndPredictions : ", filterLabelsAndPredictionsCount)

    test_data_count = float(new_test_data.count())

    testErr = filterLabelsAndPredictionsCount / test_data_count * 100

    print('Test Accuracy = ', testErr)

    print('\n\n============================ End Evaluate model ===================================\n\n')


