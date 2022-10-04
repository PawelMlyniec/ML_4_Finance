import os

import matplotlib.pyplot as plt
import pandas as pd


def describe_file(data_dir, filename):
    df = pd.read_csv(os.path.join(data_dir, filename))
    df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%dT%H:%M:%S.000Z')
    df = df.set_index("timestamp").sort_index()

    fig, axes = plt.subplots(1, 3)

    axes[0].plot(df[df.isBid == True].index, df[df.isBid == True]['price'], label='buy')
    axes[0].plot(df[df.isBid == False].index, df[df.isBid == False]['price'], label='sell')
    axes[1].plot(df[df.isBid == True].index, df[df.isBid == True]['volume'], label='buy')
    axes[1].plot(df[df.isBid == False].index, df[df.isBid == False]['volume'], label='sell')
    df.price.plot(kind='hist', ax=axes[2])

    for ax in axes:
        ax.tick_params(axis='x', labelrotation=45)
    plt.suptitle(f"{filename[:-4]}")
    plt.subplots_adjust(wspace=0.4)
    axes[0].legend()
    axes[1].legend()
    return df
