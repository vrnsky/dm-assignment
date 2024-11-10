import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def convert_to_data_frame(file_path):
    """
    Converting file to data frame
    :param file_path: path to downloaded file
    :return: data frame
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def create_technical_indicators(df):
    """
    Create technical indicators using available columns
    :param df: data frame
    :return:
    """

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    df['Daily_Return'] = df['Close'].pct_change()

    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    # Trading Range
    df['Trading_Range'] = df['High'] - df['Low']

    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Relative_Volume'] = df['Volume'] / df['Volume_MA5']

    # Price momentum (5-day)
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    # Bollinger Bands (20-day)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()

    return df


def prepare_dataset(df):
    """
    Prepare the dataset for modeling
    :param df:
    :return:
    """

    df = create_technical_indicators(df)
    df['Target'] = df['Close'].shift(-1)

    df = df.dropna()

    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'Daily_Return', 'Volatility',
        'Trading_Range', 'Volume_MA5', 'Relative_Volume',
        'Momentum', 'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]

    X = df[feature_columns]
    Y = df['Target']

    return X, Y, df


def analyze_dataset(df):
    """
    Perform initial data analysis and create visualizations
    :param df: date frame
    :return: plot
    """
    print("\nDataset Statistics:")
    print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

    # Create visualizations
    plt.figure(figsize=(15, 12))

    # Plot 1: Stock Price with Moving Averages
    ax1 = plt.subplot(3, 2, 1)
    df['Close'][-200:].plot(label='Close Price', ax=ax1)
    df['MA5'][-200:].plot(label='5-day MA', ax=ax1)
    df['MA20'][-200:].plot(label='20-day MA', ax=ax1)
    ax1.set_title('Stock Price and Moving Averages (Last 200 Days)')
    ax1.legend()

    # Plot 2: Volume Analysis
    ax2 = plt.subplot(3, 2, 2)
    df['Volume'][-100:].plot(kind='bar', ax=ax2)
    ax2.set_title('Trading Volume (Last 100 Days)')

    # Plot 3: Daily Returns Distribution
    ax3 = plt.subplot(3, 2, 3)
    df['Daily_Return'].hist(bins=50, ax=ax3)
    ax3.set_title('Distribution of Daily Returns')

    # Plot 4: Bollinger Bands
    ax4 = plt.subplot(3, 2, 4)
    df['Close'][-100:].plot(label='Close Price', ax=ax4)
    df['BB_Upper'][-100:].plot(label='Upper BB', ax=ax4)
    df['BB_Lower'][-100:].plot(label='Lower BB', ax=ax4)
    ax4.set_title('Bollinger Bands (Last 100 Days)')
    ax4.legend()

    # Plot 5: Correlation Matrix
    ax5 = plt.subplot(3, 2, 5)
    correlation_matrix = df[['Close', 'Volume', 'MA20', 'Volatility', 'Momentum']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
    ax5.set_title('Correlation Matrix of Key Features')

    # Plot 6: Trading Range
    ax6 = plt.subplot(3, 2, 6)
    df['Trading_Range'][-100:].plot(ax=ax6)
    ax6.set_title('Trading Range (High - Low) Last 100 Days')

    plt.tight_layout()
    return plt


def main():
    """
    Entry point of program. The program at the start downloading dataset from Kaggle.
    Then perform data mining and knowledge discovery operations
    :return: None
    """

    path = kagglehub.dataset_download("umerhaddii/oracle-stock-data-2024")
    print("File successfully downloaded path = ", path)
    df = convert_to_data_frame(path + "/" + "Oracle_stock.csv")
    X, Y, df = prepare_dataset(df)

    plt = analyze_dataset(df)
    plt.show()

if __name__ == '__main__':
    main()