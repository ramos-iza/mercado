import yfinance as yf
import pandas as pd 

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

def baixa_carteira_passado(nome_ativos, start_in_sample, end_in_sample): 
    carteira_passado = yf.download(nome_ativos, start = start_in_sample, end = end_in_sample)['Close']
    return carteira_passado
