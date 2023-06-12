import pandas as pd 

def adj_close(adj_close, path): 
    adj_close.to_csv(path)
    
    
def ibov(ibov, path): 
    ibov.to_csv(path)    
    
    
def carteira_passado(carteira_passado, path):
    carteira_passado.to_csv(path) 
  

