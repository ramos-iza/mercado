import silver.get as sg
from config import raw as config
import silver.transform as st
from config import otimizacao as otm 


adj_close = sg.read_csv(path=config['adj_close']['path'])

retornos_diarios = st.cal_rerornos_diarios(adj_close = adj_close)

random_pesos = st.calc_pesos(adj_close = adj_close)

retorno_portfolio_diario = st.calc_retorno_ativo(retornos_diarios = retornos_diarios, random_pesos = random_pesos)

retorno_portfolio_diario1 = st.calc_retorno_carteira_diario(retornos_diarios = retornos_diarios, random_pesos = random_pesos)

retorno_carteira_diario = st.calc_retorno_cart_diario(retorno_portfolio_diario1=retorno_portfolio_diario1)

retorno_acm_carteira = st.calc_retorno_acum_cart(retorno_carteira_diario=retorno_carteira_diario)

retorno_portfolio_anual = st.calc_retorno_port_anual(retorno_portfolio_diario=retorno_portfolio_diario)

media_retorno_portfolio_diario = st.media_retor_port_dia(retorno_portfolio_diario=retorno_portfolio_diario)

mediana_retorn_port_dia = st.mediana_retorn_port_dia(retorno_portfolio_diario=retorno_portfolio_diario)

mediana_portfolio_anual = st.calc_mediana_portfolio_anual(retorno_portfolio_anual=retorno_portfolio_anual)

shapiro_portfolio = st.calc_shapiro(retorno_carteira_diario=retorno_carteira_diario)
# com 95% de confiança há evidências estatíscas para rejeitar a hipótese nula, por tanto não é uma distribuição normal. 

skew_retorno_portfolio = st.calc_skew(retorno_carteira_diario=retorno_carteira_diario)
# Significa que a distribuição é levemente assimentrica para esquerda, se for negativa.
# Sugere que a distribuição tem uma cauda mais longa e fina à esquerda, se for nedativa. 
# Os retornos negativos são mais frequentes que os retornos positivos. No entanto, como o valor absoluto de skewness é menor do que 1, a assimetria é considerada leve. Se for anegativo. 

curtose_retorno_portfolio = st.calc_curtose(retorno_carteira_diario=retorno_carteira_diario)
# Sugere que os dados têm uma quantidade moderada de outliers ou valores extremos em relação a uma distribuição normal.
# A distribuição dos retornos é relativamente regular, sem grandes desvios em relação a uma distribuição normal.

# Medidas de disperção do Portfólio
var = st.calc_variancia(retornos_diarios=retornos_diarios)

cov = st.calc_covariancia( retornos_diarios=retornos_diarios)

cov_anual = st.calc_cov_anual(cov)

var_portfolio = st.calc_var_portfolio(random_pesos=random_pesos, cov=cov)

var_portfolio_anual = st.calc_var_anual(var_portfolio=var_portfolio)

vol_portfolio_diaria = st.calc_vol_portfolio(random_pesos=random_pesos, cov=cov)

vol_portfolio_anual = st.calc_vol_anual(random_pesos=random_pesos, cov_anual=cov_anual)

desvio_padrao = st.cal_desvio_padrao(retorno_portfolio_diario=retorno_portfolio_diario)

desvio_padrao_anual = st.calc_desvio_anual(desvio_padrao=desvio_padrao)

minimo = st.cal_min(retorno_portfolio_diario=retorno_portfolio_diario)

maximo = st.cal_max(retorno_portfolio_diario=retorno_portfolio_diario)

amplitude = st.calc_amplitude(maximo=maximo, minimo=minimo)

cv = st.calc_coef_var(retorno_portfolio_diario=retorno_portfolio_diario)

rolling_drawdown = st.calc_rolling_drawdown(retorno_carteira_diario=retorno_carteira_diario)

max_drawdown = st.calc_max_drawdown(retorno_acm_carteira=retorno_acm_carteira)

value_at_risk = st.caclc_var_at_risk(retorno_carteira_diario=retorno_carteira_diario)
# Apenas 1% dos retornos diários da carteira foram menores do que -3,1% 

media_retorno_portfolio=st.calc_media_retorno_portifolio(retorno_carteira_diario=retorno_carteira_diario)

desvio_padrao_carteira= st.calc_desvio_padrao_cart(vol_portfolio_diaria=vol_portfolio_diaria)

var_p_90 = st.cal_var_parametrico(media_retorno_portfolio=media_retorno_portfolio, desvio_padrao=desvio_padrao)
# O valor crítico (ou o limite inferior) abaixo do qual 10% dos retornos do portfólio são esperados com um nível de confiança de 90% usando a distribuição normal é -0.0813.

retorno_anualizado = st.calc_retorno_anualizado(adj_close=adj_close)

retorno_an_carteira = st.calc_retorn_an_carteira(retorno_anualizado=retorno_anualizado, random_pesos=random_pesos)

ibov = sg.read_csv(path=config['ibov']['path'])
print(ibov)

retornos_diarios_ibov = st.calc_retorno_bench(ibov=ibov)

ibov_acum = st.calc_ibov_acum(retornos_diarios_ibov=retornos_diarios_ibov)

benchmark = st.juntando_dfs(retorno_acm_carteira=retorno_acm_carteira, ibov_acum=ibov_acum)

beta_carteira = st.juntando_dfs_diarios(retornos_diarios_ibov=retornos_diarios_ibov, retorno_carteira_diario=retorno_carteira_diario)

beta_carteira = st.calc_beta(beta_carteira=beta_carteira)

sharpe_ratio = st.calc_sharpe(retornos_diarios_ibov, vol_portfolio_anual)

sortino = st.calc_sortino(retorno_carteira_diario=retorno_carteira_diario)

calmar = st.calc_calmar(retorno_carteira_diario, drawdown=max_drawdown)

carteira_futuro = sg.read_csv(path=otm['carteira_futuro']['path'])

cf_anualizado = st.calc_retorno_carteira_fut(carteira_futuro=carteira_futuro)

cf_anualizado_carteira = st.calc_retorno_ano_carteira(cf_anualizado=cf_anualizado)

carteira_futuro_retornos = st.calc_retorno_diario(carteira_futuro=carteira_futuro)

cov_carteira_futuro = st.calc_cov_cart_fut(carteira_futuro_retornos=carteira_futuro_retornos)

vol_fut_diaria = st.calc_vol_cart_fut(cov_carteira_futuro=cov_carteira_futuro)

vol_fut_ano = st.calc_vol_fut_ano(vol_fut_diaria=vol_fut_diaria)

carteira_passado = sg.read_csv(path=otm['carteira_passado']['path'])

retorno_medio = st.calc_retorno_medio(carteira_passado = carteira_passado)

ema_retorno_medio = st.calc_erro_medio_abs(retorno_medio=retorno_medio, cf_anualizado=cf_anualizado)

mme = st.calc_mme(carteira_passado=carteira_passado)

ema_mme = st.calc_erro_ema_mme(mme=mme, cf_anualizado=cf_anualizado)

selic_otm = sg.read_csv(path=otm['selic_otm']['path'])

selic_filtrada = st.filtro_datas(start_in_sample=otm['start_in_sample'], end_in_sample= otm['end_in_sample'], selic_otm=selic_otm)

selic_otm_diaria = st.calc_selic_diaria_otm(selic_filtrada=selic_filtrada)

