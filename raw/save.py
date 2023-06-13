import pandas as pd 

def adj_close(adj_close, path): 
    adj_close.to_csv(path)
    
    
def ibov(ibov, path): 
    ibov.to_csv(path)    
    
    
def carteira_passado(carteira_passado, path):
    carteira_passado.to_csv(path) 
  

def carteira_futuro(carteira_futuro, path): 
    carteira_futuro.to_csv(path) 
    
def selic_otm(selic_otm, path):
    selic_otm.to_csv(path)    
    
def ibov_sample(ibov_sample, path):
    ibov_sample.to_csv(path)    
    