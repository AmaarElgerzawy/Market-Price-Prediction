import numpy as np
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import ta

def get_linear_interpolation(src, oldMax, lookback=100):
    src = np.array(src)
    minVal = np.array([np.min(src[max(i-lookback, 0):i+1]) for i in range(len(src))])
    interpolated_values = (src - minVal) / (np.maximum(oldMax - minVal, 1e-10))  # Avoid division by zero
    return interpolated_values

def n_rsi(src, n1, n2):
    # Calculate RSI with n1 lookback period
    rsi = ta.momentum.RSIIndicator(src, window=n1).rsi()
    
    ema_rsi = rsi.ewm(span=n2, adjust=False).mean()
    
    min_val = ema_rsi.min()
    max_val = ema_rsi.max()
    linear_transformed_rsi = 100 * (ema_rsi - min_val) / (max_val - min_val)
    
    return linear_transformed_rsi

def calc_kst(src):
    # Calculate the Rate of Change (ROC) for each specified length
    roc1 = ta.momentum.ROCIndicator(src, window=10).roc()
    roc2 = ta.momentum.ROCIndicator(src, window=15).roc()
    roc3 = ta.momentum.ROCIndicator(src, window=20).roc()
    roc4 = ta.momentum.ROCIndicator(src, window=30).roc()
    
    # Apply smoothing (SMA) to the ROC values
    smoothed1 = roc1.rolling(window=3).mean()
    smoothed2 = roc2.rolling(window=3).mean()
    smoothed3 = roc3.rolling(window=3).mean()
    smoothed4 = roc4.rolling(window=3).mean()
    
    # Calculate the KST line by summing the smoothed ROC values, each multiplied by its weight
    kst_line = smoothed1 + 2 * smoothed2 + 3 * smoothed3 + 4 * smoothed4
    
    # Calculate the RSI of the KST line
    rsi_kst = ta.momentum.RSIIndicator(kst_line, window=14).rsi()
    
    return rsi_kst

def get_linear_transformation(src, min_val, max_val, lookback=200):
    # Ensure src is a pandas Series
    src = pd.Series(src)
    
    # Calculate the rolling minimum and maximum with the specified lookback period
    historic_min = src.rolling(window=lookback, min_periods=1).min()
    historic_max = src.rolling(window=lookback, min_periods=1).max()
    
    # Perform the linear transformation
    # Avoid division by zero by ensuring the denominator is at least a small number (e.g., 1e-10)
    linear_value = min_val + (max_val - min_val) * (src - historic_min) / (historic_max - historic_min).clip(lower=1e-10)
    return linear_value

def sigmoid(src,lookback=20, relative_weight=8, start_at_bar=25):
    current_weight = 0.0
    cumulative_weight = 0.0

    for i in range(1,45):
        y = src.iloc[i]
        # Calculate the weight using the provided formula
        w = np.power(1 + (np.power(i - start_at_bar, 2) / (np.power(lookback, 2) * 2 * relative_weight)), -relative_weight)
        current_weight += y * w
        cumulative_weight += w
    
    # Avoid division by zero by ensuring cumulative_weight is not zero
        sigmoid_value = current_weight / cumulative_weight if cumulative_weight != 0 else 0

    return sigmoid_value

def macd(src):
    # Calculate MACD components using the ta library
    macd_indicator = ta.trend.MACD(src, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()

# Apply linear transformation to the MACD line and signal line
    ma = get_linear_transformation(macd_line, 14, 1)
    sa = get_linear_transformation(signal_line, 14, 1)
    
    # Calculate the average of the transformed MACD and signal lines
    macd_val = (ma + sa) / 2
    
    return macd_val

def csm_cpma(data, length=21):
    price_avg = ta.trend.ema_indicator(data['close'], length)
    HL2 = (data['high'] + data['low']) / 2
    HL2_avg = HL2.rolling(window=length).mean()
    Open_avg = ta.trend.ema_indicator(data['open'], length)
    High_avg = data['high'].rolling(window=length).mean()
    Low_avg = ta.trend.ema_indicator(data['low'], length)
    OHLC4 = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    OHLC4_avg = OHLC4.rolling(window=length).mean()
    HLC3 = (data['high'] + data['low'] + data['close']) / 3
    HLC3_avg = ta.trend.ema_indicator(HLC3, length)
    HLCC4 = (data['high'] + data['low'] + data['close'] + data['close']) / 4
    HLCC4_avg = HLCC4.rolling(window=length).mean()

    # Calculate the average of the price types
    price_average = (price_avg + HL2_avg + Open_avg + High_avg + Low_avg + OHLC4_avg + HLC3_avg + HLCC4_avg) / 8
    price_average = price_average.shift(1) + (data['close'] - price_average.shift(1)) / (length * np.power(data['close']/price_average.shift(1), 4))
    
    return price_average

def series_from(feature_string, df, f_paramA, f_paramB, CPMA=None, FRMA=None):
    if feature_string == "RSI":
        return n_rsi(df['close'], f_paramA,f_paramB)
    elif feature_string == "KST":
        # Placeholder for KST calculation, as direct calculation needs custom implementation
        # Assuming calc_kst function exists and returns a Series
        kst = calc_kst(df['close'])  # Placeholder function
        return get_linear_interpolation(kst,100)
    elif feature_string == "CPMA":
        # Placeholder for CPMA calculation
        return get_linear_transformation(CPMA,14,1)
    elif feature_string == "MACD":
        return macd(df['close'])
    else:
        return pd.Series([None] * len(df))
    
class FeatureSeries:
    def __init__(self, f1, f2, f4, f6):
        self.f1 = [value for value in f1]
        self.f2 = [value for value in f2]
        self.f4 = [value for value in f4]
        self.f6 = [value for value in f6]

        tar = max(self.count_nan(self.f1),self.count_nan(self.f2),self.count_nan(self.f4),self.count_nan(self.f6))

        self.f1 = self.f1[tar:]
        self.f2 = self.f2[tar:]
        self.f4 = self.f4[tar:]
        self.f6 = self.f6[tar:]

    def count_nan(self,data):
        count = 0
        for i in data:
            if np.isnan(i):
                count+=1
        return count

class FeatureArrays:
    def __init__(self):
        self.f1 = []
        self.f2 = []
        self.f4 = []
        self.f6 = []

    def append(self, feature_series):
        self.f1.append(feature_series.f1)
        self.f2.append(feature_series.f2)
        self.f4.append(feature_series.f4)
        self.f6.append(feature_series.f6)

def get_cosine_similarity(i, featureSeries, featureArrays,time):
    dotProduct = 0.0
    magnitudeSeries = 0.0
    magnitudeArray = 0.0

    dotProduct += sum([featureSeries.f1[i] * x for x in featureArrays.f1])
    dotProduct += sum([featureSeries.f2[i] * x for x in featureArrays.f2])
    dotProduct += sum([featureSeries.f4[i] * x for x in featureArrays.f4])
    dotProduct += sum([featureSeries.f6[i] * x for x in featureArrays.f6])

    magnitudeSeries += sum([x*x for x in featureSeries.f1])
    magnitudeSeries += sum([x*x for x in featureSeries.f2])
    magnitudeSeries += sum([x*x for x in featureSeries.f4])
    magnitudeSeries += sum([x*x for x in featureSeries.f6])

    magnitudeArray += np.power(featureArrays.f1[i], 2)
    magnitudeArray += np.power(featureArrays.f2[i], 2)
    magnitudeArray += np.power(featureArrays.f4[i], 2)
    magnitudeArray += np.power(featureArrays.f6[i], 2)

    magnitudeSeries = np.sqrt(magnitudeSeries)
    magnitudeArray = np.sqrt(magnitudeArray)

    if magnitudeSeries == 0.0 or magnitudeArray == 0.0:
        return 0.0
    else:
        return dotProduct / (magnitudeSeries * magnitudeArray)
    
def get_prediction_color(prediction, isNewSellSignal):
    # Define the color array
    colors = [
        "#FF0000", "#FF1000", "#FF2000", "#FF3000", "#FF4000",
        "#FF5000", "#FF6000", "#FF7000", "#FF8000", "#FF9000",
        "#0AAA00", "#1BBB10", "#2CCC20", "#3DDD30", "#5EEE50",
        "#6FFF60", "#7ABF70", "#8BCF80", "#9CDF90", "#90DFF9"
    ]
    
    # Determine the distVal based on the condition
    if prediction >= 10 or prediction <= -10:
        distVal = -10 if isNewSellSignal else 9
    else:
        distVal = prediction
    
    # Calculate the index, ensuring it's within the bounds of the color array
    index = int(distVal + 10)
    index = max(0, min(index, len(colors) - 1))  # Ensure index is within valid range
    
    # Get the color associated with the index
    predict_color = colors[index]
    
    return predict_color, index

# Define input parameters
historyLookBack = 10000
nearest_Probable_Distance = 8

tv = TvDatafeed()
df = tv.get_hist(symbol=pair, exchange=broker, interval=period, n_bars=historyLookBack, extended_session=True)
df['close'] = [value for value in df['close'] if not np.isnan(value)]
df['open'] = [value for value in df['open'] if not np.isnan(value)]
df['low'] = [value for value in df['low'] if not np.isnan(value)]
df['high'] = [value for value in df['high'] if not np.isnan(value)]
df['volume'] = [value for value in df['volume'] if not np.isnan(value)]
dates = df.index

data = []

def core(data_frame):
    bar_index = len(data_frame['close'])
    CPMA = csm_cpma(data_frame)

    feature_series = FeatureSeries(
        series_from("CPMA", data_frame, 0, 0, CPMA=CPMA),  # f1, placeholders for actual calculation logic
        series_from("RSI", data_frame, 14, 1),  # f2
        series_from("KST", data_frame, 0, 0),   # f4
        series_from("MACD", data_frame, 0, 0)   # f6
    )

    feature_arrays = FeatureArrays()

    # Convert lists to numpy arrays for further analysis or numerical operations if needed
    feature_arrays.f1 = np.array(feature_series.f1)
    feature_arrays.f2 = np.array(feature_series.f2)
    feature_arrays.f4 = np.array(feature_series.f4)
    feature_arrays.f6 = np.array(feature_series.f6)

    maxLoob = min(len(feature_arrays.f1) ,len(feature_arrays.f2) ,len(feature_arrays.f4) ,len(feature_arrays.f6))

    trend = sigmoid(data_frame['close'])

    last_bar_index = len(data_frame['close']) - 1  # Equivalent to Pine Script's last_bar_index, assuming data_frame is indexed from 0
    maxBarsBackIndex = max(0, last_bar_index - historyLookBack)  # Translated logic

    src = list(data_frame['close'])

    y_train_array = []
    for _ in src:
        y_train_array.append(-1 if src[4] < src[0] else 1 if src[4] > src[0] else 0)
        src.pop(0)

    predictions = []
    prediction = 0.0
    signal = 0
    distances = []

    lastDistance = -1.0
    size = min(historyLookBack-1, len(y_train_array)-1)
    sizeLoop = min(historyLookBack-1, size, maxLoob)

    if bar_index >= maxBarsBackIndex:
        for i in range(sizeLoop):
            d = get_cosine_similarity(i, feature_series, feature_arrays,Interval.in_4_hour)
            if d >= lastDistance and i % 4 == 0:  # Check divisibility by 4
                lastDistance = d
                distances.append(d)
                predictions.append(round(y_train_array[i]))

                if len(predictions) > nearest_Probable_Distance:
                    lastDistance = distances[int(round(nearest_Probable_Distance * 3 / 4))]
                    distances.pop(0)  # Remove the oldest distance
                    predictions.pop(0)  # Remove the oldest prediction

        prediction = np.sum(predictions)  # Using numpy for summing list elements

    isBullishSmooth = data_frame['close'].iloc[-1] >= trend
    isBearishSmooth = data_frame['close'].iloc[-1] <= trend

    signal = np.where((prediction > 0) & (isBullishSmooth), 1, np.where((prediction < 0) & (isBearishSmooth), -1, 0))

    # Checking if the signal indicates a buy or sell
    isNewBuySignal = signal == 1
    isNewSellSignal = signal == -1

    predictColor, index = get_prediction_color(prediction,isNewSellSignal)

    return [data_frame['close'].iloc[-1],predictColor, trend]


for i in range(len(df)-200):
  data.append(core(df[i:i+200]))

cs_df = pd.DataFrame(data=data,columns=['close','color','Trend'],index=dates[200:])
cs_df.to_csv('./GradutionProject.csv')