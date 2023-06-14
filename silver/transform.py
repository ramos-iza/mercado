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
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
from pypfopt import EfficientSemivariance
from pypfopt import HRPOpt


 

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

# Calculo CAPM 
def calc_capm(carteira_passado, ibov_sample, selic_otm_diaria): 
    capm = expected_returns.capm_return(carteira_passado, market_prices = ibov_sample, risk_free_rate = selic_otm_diaria)
    return capm

# Erro médio absoluto CAPM
def calc_ema_capm(capm, cf_anualizado): 
    ema_capm = np.sum(np.abs((capm - cf_anualizado))/len(capm))
    return ema_capm


# Matriz de covariância 
def calc_sample_cov(carteira_passado): 
    sample_cov = risk_models.sample_cov(carteira_passado)
    return sample_cov


# Erro médio Sample Covariância
def calc_erro_sample_cov(sample_cov, cov_carteira_futuro): 
    ema_samble_cov = np.sum(np.abs(np.diag(sample_cov) - np.diag(cov_carteira_futuro)))/len(np.diag(sample_cov))
    return ema_samble_cov


# Semicovariância
def calc_semivar(carteira_passado):
    semi_cov = risk_models.semicovariance(carteira_passado, benchmark=0)
    return semi_cov

# Erro médio absoluto da semicovariancia 
def calc_ema_semicov(semi_cov, cov_carteira_futura, sample_cov):
    ema_semicov = np.sum(np.abs(np.diag(semi_cov) - np.diag(cov_carteira_futura)))/len(np.diag(sample_cov))
    return ema_semicov


# Exponentially-Weighted Covariance
def calcl_exp_cov(carteira_passado): 
    exp_cov = risk_models.exp_cov(carteira_passado, span=200)
    return exp_cov

# Erro médio Exponentially-Weighted Covariance
def ema_exp_cov(exp_cov, cov_carteira_futuro):
    ema_exp_cov = np.sum(np.abs(np.diag(exp_cov) - np.diag(cov_carteira_futuro)))/len(np.diag(exp_cov))
    return ema_exp_cov

# Ledoit Wolf
def calc_lq_cov(carteira_passado):
    lw_cov = risk_models.CovarianceShrinkage(carteira_passado).ledoit_wolf()
    return lw_cov

# Erro médio absoluto Ledoit Wolf cov 
def calc_ema_lw_cov(lw_cov,cov_carteira_futuro):
    ema_lw_cov = np.sum(np.abs(np.diag(lw_cov) - np.diag(cov_carteira_futuro)))/len(np.diag(lw_cov))
    return ema_lw_cov

# Otimização 

# Miníma variância 
def otm_mv(capm,semi_cov):
    mv = EfficientFrontier(capm, semi_cov)
    return mv

# Pesos otimização miníma variância 
def pesos_min_vol(mv):
    mv.min_volatility()
    pesos_vol = mv.clean_weights()
    return pesos_vol

def selic_otm_aa(selic_filtrada):
    selic_otm_aa = selic_filtrada.Value.mean()/100
    return selic_otm_aa


# Performance da otimização de miníma variância 
def perf_mv(mv, selic_otm_aa): 
    perf_mv = mv.portfolio_performance(verbose = True, risk_free_rate = selic_otm_aa)
    return perf_mv


# Função Regularizadora
def otm_funcao_regularizadora(capm, semi_cov): 
    mv_2 = EfficientFrontier(capm, semi_cov)
    mv_2.add_objective(objective_functions.L2_reg, gamma=0.1)
    mv_2.min_volatility()
    return mv_2

def pesos_funcao_regularizadora(mv_2): 
    pesos_2 = mv_2.clean_weights()
    pesos_2 = pesos_2.values()
    pesos_2 = list(pesos_2)
    pesos_2 = np.array(pesos_2)
    return pesos_2

def otm_vol_funcao_regularizadora(pesos_2, cov_carteira_futuro): 
    vol_otimizada_2 = np.sqrt(np.dot(pesos_2.T, np.dot(cov_carteira_futuro,pesos_2)))
    vol_otimizada_2 = vol_otimizada_2*np.sqrt(252)
    return vol_otimizada_2

def calc_retorno_min_vol(cf_anualizado, pesos_2): 
    retorno_min_vol2 = cf_anualizado.dot(pesos_2)
    return retorno_min_vol2


def perf_mv2(selic_otm_aa, mv_2):
    perf_mv_2 = mv_2.portfolio_performance(verbose = True, risk_free_rate = selic_otm_aa)
    return perf_mv_2


# Risco Eficiente
def otm_risco_eficiente(capm, semi_cov):
    risco_eficiente = EfficientFrontier(capm, semi_cov)
    risco_eficiente.efficient_risk(target_volatility=0.25)
    return risco_eficiente


def pesos_re(risco_eficiente): 
    re_pesos = risco_eficiente.clean_weights(rounding = 2)
    re_pesos = re_pesos.values()
    re_pesos = np.array(list(re_pesos))
    return re_pesos


def perf_re(risco_eficiente, selic_otm_aa): 
    perf_re = risco_eficiente.portfolio_performance(verbose=True, risk_free_rate=selic_otm_aa)
    return perf_re


def otm_vol_re(re_pesos, cov_carteira_futura):
    vol_re_otimizada = np.sqrt(np.dot(re_pesos.T, np.dot(cov_carteira_futura, re_pesos)))
    vol_re_otimizada = vol_re_otimizada * np.sqrt(252)
    return vol_re_otimizada 
    

def calc_retorno_re(cf_anualizado, re_pesos):   
    retorno_re_otimizado = cf_anualizado.dot(re_pesos)
    return retorno_re_otimizado    


# Retorno Eficiente
def calc_retorno_eficiente(capm, semi_cov):
    retorno_eficiente = EfficientFrontier(capm, semi_cov)
    retorno_eficiente.efficient_return(target_return=0.0)
    return retorno_eficiente


def pesos_retorno_eficiente(retorno_eficiente):
    pesos_retorno_eficiente = retorno_eficiente.clean_weights()
    pesos_retornos_eficiente = pesos_retorno_eficiente.values()
    pesos_retornos_eficiente = np.array(list(pesos_retornos_eficiente))
    return pesos_retornos_eficiente


def calc_retorno_eficiente_2(cf_anualizado, pesos_retornos_eficiente):
    retorno_eficiente_2 = cf_anualizado.dot(pesos_retornos_eficiente)
    return retorno_eficiente_2

# Hierarchical Risk Parity
def calc_retorno_rp(carteira_passado):
    retornos_rp = expected_returns.returns_from_prices(carteira_passado)
    retornos_rp = retornos_rp.dropna()
    return retornos_rp

def hrp_portfolio(retornos_rp):
    hrp_portfolio = HRPOpt(retornos_rp)
    hrp_portfolio.optimize()
    return hrp_portfolio

def perf_hrp(hrp_portfolio, selic_otm_aa):
    perf_hrp = hrp_portfolio.portfolio_performance(verbose=True, risk_free_rate=selic_otm_aa)
    return perf_hrp

# Max Sharpe 
# A otimização de Max Sharpe foi impossibilitada porque o retorno de nenhum dos ativos é maior do que o retorno da taxa livre de risco.
def otm_max_sharpe(capm, semi_cov): 
    msharpe = EfficientFrontier(capm, semi_cov)
    msharpe.max_sharpe(risk_free_rate=selic_otm_aa)
    return msharpe


def pesos_msharpe(msharpe):
    pesos_msharpe = msharpe.clean_weights()
    pesos_msharpe = pesos_msharpe.values =()
    pesos_msharpe = list(pesos_msharpe)
    pesos_msharpe = np.array(pesos_msharpe)
    return pesos_msharpe


def otm_vol_msharpe(pesos_msharpe, cov_carteira_futura):
    volsharpe = np.sqrt(np.dot(pesos_msharpe.T, np.dot(cov_carteira_futura, pesos_msharpe)))
    volsharpe = volsharpe * np.sqrt(252)
    return volsharpe


def retorno_msharpe(cf_anualizado, sharpe_pesos): 
    retorno_sharpe = cf_anualizado.dot(sharpe_pesos)
    return retorno_sharpe









    


    






    

