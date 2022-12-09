import numpy as np
import pandas as pd
import time
pd.options.plotting.backend = "plotly"

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import Callback

import Utils
from DataFrameOnSteroids import DataFrameOnSteroids
import tensorflow as tf
import sklearn.metrics as sklearnMetrics
from functools import wraps
import os

from Utils import track

def getSomeRegressionMetrics():
    metrics = [
        sklearnMetrics.explained_variance_score,
        sklearnMetrics.max_error,
        sklearnMetrics.mean_absolute_error,
        sklearnMetrics.mean_squared_error,
        sklearnMetrics.mean_squared_error,
        sklearnMetrics.mean_squared_log_error,
        sklearnMetrics.median_absolute_error,
        sklearnMetrics.r2_score,
        #sklearnMetrics.mean_poisson_deviance,
        #sklearnMetrics.mean_gamma_deviance,
        sklearnMetrics.mean_absolute_percentage_error,
        #sklearnMetrics.d2_absolute_error_score,
        #sklearnMetrics.d2_pinball_score,
        #sklearnMetrics.d2_tweedie_score,
    ]
    return {metricFunc.__name__: metricFunc for metricFunc in metrics}

def addFullyConnectedLayers(model, layerCount, inputSize, outputSize, modelThickness):
    if len(model.layers):
        layerCount = layerCount
        modelThickness = modelThickness
        model.add(Dense(modelThickness, activation='relu'))
    else:
        modelThickness = modelThickness if modelThickness else int(1* inputSize)
        layerCount = layerCount if layerCount else int(1 * inputSize)
        model.add(Dense(modelThickness, input_shape=(inputSize,), activation='relu'))
    # tf.keras.layers.Dropout(.9, input_shape=(inputSize,))

    for i in range(layerCount - 2):
        model.add(Dense(modelThickness, activation='relu'))
        # tf.keras.layers.Dropout(.9, input_shape=(modelThickness,))

    model.add(Dense(outputSize, activation='relu'))

def buildANN(fullyConnectedLayerCount, inputSize, outputSize, customAddConvLayers, modelThickness) -> Sequential:
    model = Sequential()
    if customAddConvLayers:
        customAddConvLayers(model)
    addFullyConnectedLayers(model, fullyConnectedLayerCount, inputSize, outputSize, modelThickness)
    return model

class ExpectedScoresPerEpoch(Callback):
    def __init__(self, expectedDeadlines:dict[int,float], monitor='loss'):
        super().__init__()
        self.monitor = monitor
        self.expectedDeadlines = expectedDeadlines
        self.epochDeadline = 10
        self.epochCount = 0

    def on_epoch_end(self, batch, logs):
        score = logs.get(self.monitor)
        self.epochCount += 1
        if self.epochCount in self.expectedDeadlines and score > self.expectedDeadlines[self.epochCount]:
            self.model.stop_training = True
            print("Deadline Not Met!")
            raise Exception("Deadline Not Met!")

class StopAtStagnation(Callback):
    def __init__(self, monitor='loss', epochCount=10, decimals=4):
        super().__init__()
        self.monitor = monitor
        self.prevEpochsScores = []
        self.epochCount = epochCount
        self.decimals = decimals

    def on_epoch_end(self, batch, logs):
        score = logs.get(self.monitor)
        self.prevEpochsScores.append(round(score,self.decimals))
        self.prevEpochsScores = self.prevEpochsScores[-self.epochCount:]
        print(f"\nlast {len(self.prevEpochsScores)} epochs' {self.monitor}: {[round(i,self.decimals) for i in self.prevEpochsScores]}")
        if len(self.prevEpochsScores) == self.epochCount and len(set(self.prevEpochsScores)) == 1:
            self.model.stop_training = True
            print("Stagnation Detected!")

class ModelOnSteroids:
    def __init__(self, df: DataFrameOnSteroids=None, fullyConnectedLayerCount=None, model=None, xNames=None, yNames=None,
                 trainMetric="mean_squared_error", inputSize=None, customAddConvLayers=None, modelThickness=None):
        self.rawModel = model
        self.xNames = xNames
        self.yNames = yNames
        self.trainMetric = trainMetric
        self.callbacks = []
        if self.rawModel is None:
            rawModel = buildANN(fullyConnectedLayerCount, inputSize=inputSize if inputSize else len(df.xNames),
                outputSize=len(df.yNames), customAddConvLayers=customAddConvLayers, modelThickness=modelThickness)
            rawModel.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=[self.trainMetric])
            print(rawModel.summary())
            self.rawModel = rawModel
            self.xNames = df.xNames
            self.yNames = df.yNames
        pass

    @track
    def train(self, df: DataFrameOnSteroids, batch_size=None, epochs=5, deadlines=None,
              trainHistoryOutFileName="output\\trainHistory.pkl"):
        df.getNonBaseState()
        batch_size = batch_size if batch_size else min(df.shape[0], 256)

        self.callbacks.append(StopAtStagnation(monitor=f'val_{self.trainMetric}'))
        if deadlines:
            self.callbacks.append(ExpectedScoresPerEpoch(deadlines, monitor=f'{self.trainMetric}'))

        x_train, y_train = df.dfTrain.x.to_numpy(), df.dfTrain.y.to_numpy()
        x_val, y_val = df.dfTest.x.to_numpy(), df.dfTest.y.to_numpy()
        print(f"\nModel Input:\n{df.xNames}\n\nModel Output:\n{df.yNames}\n\nDataFrame:\n{df}\nbatch_size: {batch_size}")
        t = time.time()
        history = self.rawModel.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                                    callbacks=self.callbacks)
        print(f"Training Duration: {(time.time()-t)/60:.2f} minutes")

        if trainHistoryOutFileName:
            df = pd.DataFrame(data=history.history)
            df.index.name = "epoch"
            Utils.saveOnDisk(df, trainHistoryOutFileName)
            print();print(df);print()
            relevantCols = [self.trainMetric, f"val_{self.trainMetric}"]
            fig = df[relevantCols].plot(title="Model Training History", template="simple_white",
                          labels=dict(index="epoch", value="value", variable="metric"))
            fig.write_image(f"{trainHistoryOutFileName}.png")
        return history

    @track
    def predict(self, x: np.ndarray):
        return self.rawModel.predict(x)

    @track
    def test(self, df: DataFrameOnSteroids, metrics=None, metricResultsOutFileName="output\\metricResults.pkl"):
        if metrics is None:
            metrics = getSomeRegressionMetrics()

        y_pred = self.predict(x=df.dfTest.x.to_numpy())
        y_true = df.dfTest.y.to_numpy()

        metricResults = {metricName: metricFunc(y_true, y_pred) for metricName, metricFunc in metrics.items()}

        pd.options.display.float_format = '{:.2f}'.format
        dfMetricResults = pd.DataFrame(data={"metric": metricResults.keys(), "score": metricResults.values()})
        print("\n",dfMetricResults,"\n")
        if metricResultsOutFileName:
            Utils.saveOnDisk(dfMetricResults, metricResultsOutFileName)
        return metricResults

def loadModelFromDisk(fileName:str):
    fileXNames = f'{fileName}.xNames.pkl'
    fileYNames = f'{fileName}.yNames.pkl'
    xNames = Utils.loadFromDisk(fileXNames)
    yNames = Utils.loadFromDisk(fileYNames)
    rawModel = load_model(fileName)
    model = ModelOnSteroids(model=rawModel, xNames=xNames, yNames=yNames)
    return model

def writeModelOnDisk(fileName:str, model:ModelOnSteroids):
    fileXNames = f'{fileName}.xNames.pkl'
    fileYNames = f'{fileName}.yNames.pkl'
    model.rawModel.save(fileName)
    Utils.saveOnDisk(model.xNames, fileXNames)
    Utils.saveOnDisk(model.yNames, fileYNames)

def modelOnSteroidsIsStorableOnDiskForFutureFunctionCalls(function_to_decorate):
    @wraps(function_to_decorate)
    def wrapper(*args, **kwargs):
        fileName = f"output\\{function_to_decorate.__name__}.kerasModel"
        toLoadFromDisk = os.path.exists(fileName)
        model = loadModelFromDisk(fileName) if toLoadFromDisk else function_to_decorate(*args, **kwargs)
        if not toLoadFromDisk:
            writeModelOnDisk(fileName, model)

        return model
    return wrapper