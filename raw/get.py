import yfinance as yf
import pandas as pd 
import nasdaqdatalink 

def baixar_carteira(inicio, fim): 
    ativos = ['B3SA3', 'AGRO3', 'COCA34', 'CPLE6', 'TAEE11', 'VALE3', 'CYRE3']
    nome_ativos = pd.Series(ativos)
    nome_ativos = (nome_ativos + '.SA').tolist()
    
    adj_close = yf.download(tickers = nome_ativos,   start = inicio, end = fim, rounding = True)['Adj Close']
    return adj_close


def baixar_dados_bench(inicio, fim): 
    ibov = yf.download(tickers= '^BVSP', start= inicio, end= fim)['Adj Close']
    ibov = pd.DataFrame(ibov)
    return ibov


def baixar_carteira_passado(nome_ativos, start_in_sample, end_in_sample): 
    carteira_passado = yf.download(nome_ativos, start = start_in_sample, end = end_in_sample)['Close']
    return carteira_passado


def baixar_carteira_futuro(nome_ativos, start_out_sample, end_out_sample):
    carteira_futuro = yf.download(nome_ativos, start = start_out_sample, end = end_out_sample)['Close']
    return carteira_futuro


def baixar_selic_otimizada(start_in_sample,end_out_sample):
    selic_otm = nasdaqdatalink.get('BCB/432', start_date=start_in_sample, end_date=end_out_sample, collapse='daily')
    return selic_otm


def baixar_ibov_in_sample(start_in_sample, end_in_sample): 
    ibov_sample = yf.download('^BVSP', start = start_in_sample, end = end_in_sample)['Close']
    ibov_sample = pd.DataFrame(ibov_sample)
    ibov_sample
    return ibov_sample

