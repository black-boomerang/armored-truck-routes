import numpy as np
import pandas as pd


def create_report(data, terminals, filename):
    data = np.stack(data, axis=1)
    df = pd.DataFrame()
    df['устройство'] = terminals['TID']

    for i in range(data.shape[1]):
        df[f'day {i + 1}'] = data[:, i]

    df.to_csv(f'./reports/{filename}.csv', index=False, encoding='utf-8')


def create_final_report(cashable, non_cashable, N=10):
    cashable = np.sum(cashable, axis=0)
    non_cashable = np.sum(non_cashable, axis=0)
    total = cashable + non_cashable + N * 20000

    df = pd.DataFrame()
    df['статья расходов'] = ['фондирование', 'инкассация', 'стоимость броневиков', 'итого']
    
    for i in range(len(total)):
        df[f'day {i + 1}'] = np.array([cashable[i], non_cashable[i], N * 20000, total[i]])

    df.to_csv(f'./reports/итог.csv', index=False, encoding='utf-8')
