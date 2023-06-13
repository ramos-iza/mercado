import pandas as pd 
from config import raw as config
import raw.get as rg
import raw.save as rs
from config import otimizacao as otm 
import nasdaqdatalink

adj_close = rg.baixar_carteira(config['inicio'], config['fim'])

rs.adj_close(adj_close=adj_close,
             path=config['adj_close']['path'])

ibov = rg.baixar_dados_bench(config['inicio'], config['fim'])

rs.ibov(ibov=ibov,
        path=config['ibov']['path'])

nome_ativos = otm['nome_ativos']
nome_ativos = [ativo + '.SA' for ativo in nome_ativos]
    
carteira_passado = rg.baixar_carteira_passado(nome_ativos=nome_ativos, start_in_sample=otm['start_in_sample'], end_in_sample=otm['end_in_sample'])

rs.carteira_passado(carteira_passado=carteira_passado,
                    path=otm['carteira_passado']['path'])

carteira_futuro = rg.baixar_carteira_futuro(nome_ativos=nome_ativos, start_out_sample=otm['start_out_sample'], end_out_sample=otm['end_out_sample'])

rs.carteira_futuro(carteira_futuro=carteira_futuro,
                   path=otm['carteira_futuro']['path'])

nasdaqdatalink.ApiConfig.api_Key = otm['selic_otm']['key']
selic_otm = rg.baixar_selic_otimizada(start_in_sample=otm['start_in_sample'], end_out_sample=otm['end_out_sample'])

rs.selic_otm(selic_otm=selic_otm, path=otm['selic_otm']['path'])

ibov_sample = rg.baixar_ibov_in_sample(start_in_sample=otm['start_in_sample'], end_in_sample=otm['end_in_sample'])

rs.ibov_sample(ibov_sample=ibov_sample, path=otm['ibov_sample']['path'])

