import silver.get as sg
from config import raw as config
import numpy as np 
import pandas as pd
from scipy.stats import shapiro
from scipy import stats
import pylab 
from scipy.stats import skew
from scipy.stats import kurtosis 
from scipy.stats import norm
import statsmodels.api as sm
from pypfopt import expected_returns

 

# Calcular os retornos diários 
def cal_rerornos_diarios(adj_close):
    retornos_diarios = adj_close.pct_change()
    return retornos_diarios


# Multiplicação matricial 
# Usando pesos definidos randomicamente sem qualquer modelo de otimização
# Usei a seed 42 para sempre repetir o mesmo peso e facilitar as análise anteriores

def calc_pesos(adj_close):
    np.random.seed(42)
    random_pesos = np.random.random(len(adj_close.columns))
    random_pesos = random_pesos/sum(random_pesos)
    return random_pesos


# Mediadas descritivas do Portfólio - desempenho

# Por ativo 
def calc_retorno_ativo(retornos_diarios, random_pesos): 
    retorno_portfolio_diario = (retornos_diarios * random_pesos).sum()
    return retorno_portfolio_diario.sort_values()


# Da carteira 
def calc_retorno_carteira_diario(retornos_diarios, random_pesos): 
    retorno_portfolio_diario1 = (retornos_diarios * random_pesos).sum(axis=1)
    return retorno_portfolio_diario1


# Retorno da carteira diário 
def calc_retorno_cart_diario(retorno_portfolio_diario1):
    retorno_carteira_diario = pd.DataFrame()
    retorno_carteira_diario['Retornos'] = retorno_portfolio_diario1
    return retorno_carteira_diario
    
    
# Retorno acumulado da carteira     
def calc_retorno_acum_cart(retorno_carteira_diario):
    retorno_acm_carteira = (1 + retorno_carteira_diario).cumprod()
    return retorno_acm_carteira


# Retorno Portifólio anual
def calc_retorno_port_anual(retorno_portfolio_diario):
    retorno_portfolio_anual = retorno_portfolio_diario.mean()*252
    return retorno_portfolio_anual


# Média dos retornos diários do portfolio 
def media_retor_port_dia(retorno_portfolio_diario):
    media_retorno_portfolio_diario = retorno_portfolio_diario.mean()
    return media_retorno_portfolio_diario


# Mediana dos retornos diários do portfolio 
def mediana_retorn_port_dia(retorno_portfolio_diario):
    mediana_retorno_portfolio_diario = retorno_portfolio_diario.median()
    return mediana_retorno_portfolio_diario


# Mediana dos retornos anuiais do portfolio
def calc_mediana_portfolio_anual(retorno_portfolio_anual):
    mediana_portfolio_anual = np.median(retorno_portfolio_anual)
    return mediana_portfolio_anual


# Teste de normalidade dos retornos do portfolio 
def calc_shapiro(retorno_carteira_diario):
    shapiro_portfolio = shapiro(retorno_carteira_diario)
    return shapiro_portfolio


# Assimetria da distribuição
def calc_skew(retorno_carteira_diario):
    skew_retorno_portfolio = skew(retorno_carteira_diario)
    return skew_retorno_portfolio


# Curtose 
def calc_curtose(retorno_carteira_diario): 
    curtose_retorno_portfolio = kurtosis(retorno_carteira_diario)
    return curtose_retorno_portfolio


# Calculando a variâcia 
def calc_variancia(retornos_diarios):
    var = retornos_diarios.var()
    var = var.sort_values()
    return var


# Calculando a covariancia do portfólio 
def calc_covariancia(retornos_diarios):
    cov = retornos_diarios.cov()
    return cov


# Calculando a covariancia anual do portifólio
def calc_cov_anual(cov): 
    cov_anual = cov * 252
    return cov_anual 


# Calculando a variância do portfólio 
def calc_var_portfolio(random_pesos, cov):
    var_portfolio = (random_pesos.dot(cov)).dot(random_pesos)
    return var_portfolio


# Calculando a variância anual do portfólio  
def calc_var_anual(var_portfolio):
    var_portfolio_anual = var_portfolio.mean() * 252 
    return var_portfolio_anual


# Volatilidade do portfolio 
def calc_vol_portfolio(random_pesos, cov):
    vol_portfolio_diaria = np.sqrt(np.dot(random_pesos.T, np.dot(cov, random_pesos)))
    return vol_portfolio_diaria


# Volatilidade anual do portfolio
def calc_vol_anual(random_pesos, cov_anual):
    vol_portfolio_anual = np.sqrt(np.dot(random_pesos.T, np.dot(cov_anual, random_pesos)))
    return vol_portfolio_anual


#Desvio padrão 
def cal_desvio_padrao(retorno_portfolio_diario):
    desvio_padrao = retorno_portfolio_diario.std()
    return desvio_padrao


# Desvio padrão 
def calc_desvio_anual(desvio_padrao): 
    desvio_padrao_anual = desvio_padrao.mean()*252
    return desvio_padrao_anual

# Medidas de amplitude
def cal_min(retorno_portfolio_diario): 
    minimo = retorno_portfolio_diario.min()
    return minimo 

def cal_max(retorno_portfolio_diario): 
    maximo = retorno_portfolio_diario.max()
    return maximo

def calc_amplitude(maximo, minimo): 
    amplitude = maximo - minimo
    return amplitude 

#Coeficiente de variacao 
def calc_coef_var(retorno_portfolio_diario): 
    cv = (retorno_portfolio_diario.std()/retorno_portfolio_diario.mean())
    return cv


# Rolling drawdown 
def calc_rolling_drawdown(retorno_carteira_diario):
    rolling_drawdown = retorno_carteira_diario.rolling(window=20).min()
    return rolling_drawdown


# Max drawdown 
def calc_max_drawdown(retorno_acm_carteira):
    pico = retorno_acm_carteira.expanding(min_periods=1).max()
    dd = (retorno_acm_carteira/pico) - 1
    drawdown = dd.min()
    return drawdown

# Value at Risk 
def caclc_var_at_risk(retorno_carteira_diario):
    var_95 = np.percentile(retorno_carteira_diario, 5)
    var_98 = np.percentile(retorno_carteira_diario, 2)
    var_99 = np.percentile(retorno_carteira_diario, 1)
    return var_95, var_98, var_99


# Parametros da amostra 
def calc_media_retorno_portifolio(retorno_carteira_diario): 
    media_retorno_portfolio = np.mean(retorno_carteira_diario)
    return media_retorno_portfolio
 

def calc_desvio_padrao_cart(vol_portfolio_diaria):    
    desvio_padrao_carteira = vol_portfolio_diaria
    return desvio_padrao_carteira

# Var paramétrico 
def cal_var_parametrico(media_retorno_portfolio, desvio_padrao): 
    var_p_90 = norm.ppf(1-0.9, media_retorno_portfolio, desvio_padrao)
    return var_p_90


# Retorno Anualizado 
def calc_retorno_anualizado(adj_close):
    retorno_anualizado = (adj_close.iloc[-1] - adj_close.iloc[0])/(adj_close.iloc[0])
    return retorno_anualizado

#Retorno anualizado carteira 
def calc_retorn_an_carteira(retorno_anualizado, random_pesos): 
    retorno_an_carteira = ((1 + retorno_anualizado)**(12/24))-1
    retorno_an_carteira = retorno_an_carteira.dot(random_pesos)
    return retorno_an_carteira


# Retorno benchmark 
def calc_retorno_bench(ibov): 
    retornos_diarios_ibov = ibov.pct_change()
    return retornos_diarios_ibov


# Retorno do bench acumulado 
def calc_ibov_acum(retornos_diarios_ibov):
    ibov_acum = (retornos_diarios_ibov + 1).cumprod()
    ibov_acum.rename(columns={'Adj Close':'Ibov'},inplace=True)
    return ibov_acum

# Juntando os Dataframes 
def juntando_dfs(retorno_acm_carteira, ibov_acum): 
    benchmark = pd.merge(retorno_acm_carteira, ibov_acum, how= 'inner', right_index=True, left_index=True).dropna()
    return benchmark

# Juntando dataframes para calcular o beta 
def juntando_dfs_diarios(retornos_diarios_ibov, retorno_carteira_diario): 
    retornos_diarios_ibov.rename(columns={'Adj Close' : 'Ibov'}, inplace=True)
    beta_carteira = pd.merge(retorno_carteira_diario, retornos_diarios_ibov, how = 'inner', left_index=True, right_index=True).dropna()
    return beta_carteira

# Didicar variáveis 
# x independente e y dependente 
def calc_beta(beta_carteira):
    y = beta_carteira['Retornos']
    x = beta_carteira['Ibov']

    x = sm.add_constant(x)

    modelo = sm.OLS(x,y)
    resultado = modelo.fit()
    return resultado.params[1]


# Calculo sharpe e sortino 
selic = 0.0905
def calc_sharpe(retornos_diarios_ibov, vol_portfolio_anual): 
    vol_ibov_anual = retornos_diarios_ibov.std() * np.sqrt(252)

    sharpe_ratio_ibov = ((retornos_diarios_ibov.mean()*252)-(selic)/(vol_portfolio_anual))
    return sharpe_ratio_ibov   

def calc_sortino(retorno_carteira_diario):
    sortino = ((retorno_carteira_diario.mean()*252) - (retorno_carteira_diario[(retorno_carteira_diario<0)]).std()*np.sqrt(252))
    return sortino     


# Calmar 
# A métrica Calmar indica o retorno anualizado que a carteira gera em excesso em relação à taxa de juros livre de risco (SELIC), dividido pela magnitude da maior perda da carteira em relação ao máximo pico.
# Em outras palavras, a carteira gerou um retorno anualizado que excedeu a taxa de juros livre de risco em cerca de 29% em relação ao drawdown observado. 

def calc_calmar(retorno_carteira_diario, drawdown):
    calmar = ((retorno_carteira_diario.mean()*252)-selic)/abs(drawdown)
    return calmar


# Retorno anualizado da carteira futura
pesos = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
def calc_retorno_carteira_fut(carteira_futuro): 
    cf_anualizado = (carteira_futuro.iloc[-1] - carteira_futuro.iloc[0])/carteira_futuro.iloc[0]
    cf_anualizado = ((1+ cf_anualizado) ** (12/48))-1
    return cf_anualizado


def calc_retorno_ano_carteira(cf_anualizado):
    cf_anualizado_carteira = cf_anualizado.dot(pesos)
    return cf_anualizado_carteira


# Retorno diário da carteira futuro 
def calc_retorno_diario(carteira_futuro):
    carteira_futuro_retornos = carteira_futuro.pct_change()
    return carteira_futuro_retornos


# Covariância da carteira Futura 
def calc_cov_cart_fut(carteira_futuro_retornos): 
    cov_carteira_futuro = carteira_futuro_retornos.cov()
    return cov_carteira_futuro


# Volatilidade da carteira Futura
def calc_vol_cart_fut(cov_carteira_futuro):
    vol_fut_diaria = np.sqrt(np.dot(pesos.T, np.dot(cov_carteira_futuro, pesos)))
    return vol_fut_diaria


# Volatilidade da carteira Futura ao ano 
def calc_vol_fut_ano(vol_fut_diaria):
    vol_fut_ano = vol_fut_diaria*np.sqrt(252)
    return vol_fut_ano


# Estimando os retornos 
def calc_retorno_medio(carteira_passado): 
    retorno_medio = expected_returns.mean_historical_return(carteira_passado)
    return retorno_medio

# EMA - erro médio absoluto
def calc_erro_medio_abs(retorno_medio, cf_anualizado): 
    ema_retorno_medio = (np.sum(np.abs(retorno_medio - cf_anualizado))/len(retorno_medio))
    return ema_retorno_medio

# MME - Média Móvel Exponencial
def calc_mme(carteira_passado):
    mme = expected_returns.ema_historical_return(carteira_passado , span=200)
    return mme

# Erro médio absoluto da média móvel exponencial
def calc_erro_ema_mme(mme, cf_anualizado):
    ema_mme = np.sum(np.abs(mme-cf_anualizado))/len(mme)
    return ema_mme

# CAPM 

# Taxa livre de risco 

def filtro_datas(start_in_sample, end_in_sample, selic_otm): 
    selic_otm.index = pd.to_datetime(selic_otm.index)
    data_range = pd.date_range(start_in_sample, end_in_sample)
    selic_filtrada = selic_otm[selic_otm.index.isin(data_range)]
    return selic_filtrada

def calc_selic_diaria_otm(selic_filtrada):
    selic_filtrada_copia = selic_filtrada.copy()
    selic_filtrada_copia['selic_otm_ad'] = ((1+(selic_filtrada_copia.Value/100))**(1/252)-1)
    selic_otm_diaria = selic_filtrada_copia.selic_otm_ad.mean()
    return selic_otm_diaria









    


    






    

