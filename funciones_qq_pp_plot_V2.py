# -*- coding: utf-8 -*-
"""
**Librerías**
"""

import math
import numpy as np
import statistics as st
from scipy.stats import expon, norm, lognorm, gamma, weibull_min, beta, uniform, chi2, triang
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

"""**Gráficas PP plot y QQ plot para una distribución normal**"""
def PP_QQ_plot_normal(data,media="estimado",desvesta="estimado", ax1=None, ax2=None):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(data)
    else:
        mean = media
    if desvesta=="estimado":
        std_dev = np.std(data)
    else:
        std_dev = desvesta
    
    n = len(data)
    # Se crea una Q-Q plot con el paquete statsmodels.api para la distribución normal
    sm.qqplot(data, norm, loc=mean, scale=std_dev, line='45', ax= ax1)
    # Se agrega un título a la gráfica
    #plt.title("Q-Q Plot Normal")
    ax1.set_title("Q-Q Plot Normal")
   
    
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(norm.cdf(data,loc=mean,scale=std_dev))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Normal')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    return (mean, std_dev)
    
"""**Gráficas PP plot y QQ plot para una distribución lognormal**"""
def PP_QQ_plot_lognormal(data,media="estimado",desvesta="estimado",ax1=None, ax2=None):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(np.log(data))
    else:
        mean = media
    
    if desvesta=="estimado":
        std_dev = np.std(np.log(data))
    else:
        std_dev = desvesta  
    
    n = len(data)
    sm.qqplot(data, lognorm, distargs=(std_dev,), scale=np.exp(mean), line='45', ax=ax1)
    # Se agrega un título a la gráfica
    ax1.set_title("Q-Q Plot Lognormal")
   
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(lognorm.cdf(data,s=std_dev,scale=np.exp(mean)))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Lognormal')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
  
    return (mean, std_dev)
    

"""**Gráficas PP plot y QQ plot para una distribución exponencial**"""
def PP_QQ_plot_exponential(data,tasa="estimado",ax1=None, ax2=None):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if tasa=="estimado":
        mean = np.mean(data)
    else:
        mean = 1/tasa
    
    n = len(data)
    sm.qqplot(data, expon, scale=mean, line='45', ax=ax1)
    # Se agrega un título a la gráfica
    ax1.set_title("Q-Q Plot Exponencial")
   
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(expon.cdf(data,scale=mean))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Exponencial')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    return (mean)
    

"""**Gráficas PP plot y QQ plot para una distribución uniforme**"""
def PP_QQ_plot_uniform(data,minimo="estimado",maximo="estimado",ax1=None, ax2=None):
    n = len(data)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if minimo=="estimado":
        a = np.min(data)
    else:
        a = minimo
    
    if maximo=="estimado":
        b = np.max(data)
    else:
        b = maximo

    sm.qqplot(data,uniform,loc=a,scale=b-a,line='45', ax=ax1)
    # Se agrega un título a la gráfica
    ax1.set_title("Q-Q Plot Uniforme")
    # Se muestra la gráfica
    
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(uniform.cdf(data,loc=a,scale=b-a))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Uniforme')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    return (a, b)
    

"""**Gráficas PP plot y QQ plot para una distribución triangular**"""
def PP_QQ_plot_triangular(data,minimo="estimado",maximo="estimado",moda="estimado",ax1=None, ax2=None):
    n = len(data)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if minimo=="estimado":
        a = np.min(data)
    else:
        a = minimo

    if maximo=="estimado":
        b = np.max(data)
    else:
        b = maximo
        
    if moda=="estimado":
        c = st.mode(data)
    else:
        c = moda
    
    sm.qqplot(data,triang,distargs=((c - a)/(b - a),),loc=a,scale=b-a,line='45', ax=ax1)
    # Se agrega un título a la gráfica
    ax1.set_title("Q-Q Plot Triangular")
    
        # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(triang.cdf(data,c=(c - a)/(b - a),loc=a,scale=b-a))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Triangular')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    return (a, c, b)

"""**Gráficas PP plot y QQ plot para una distribución gamma**"""
def PP_QQ_plot_gamma(data,media="estimado",varianza="estimado",ax1=None, ax2=None):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(data)
    else:
        mean = media
    
    if varianza=="estimado":
        var = np.var(data)
    else:
        var = varianza
    
    n = len(data)
    sm.qqplot(data,gamma,distargs=(var/mean,),scale=mean,line='45', ax=ax1)
    # Se agrega un título a la gráfica
    ax1.set_title("Q-Q Plot Gamma")
   
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(gamma.cdf(data,a=var/mean, scale=mean))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Gamma')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
   
    return (mean,var)

"""**Gráficas PP plot y QQ plot para una distribución weibull**"""
def PP_QQ_plot_weibull(data,forma="estimado",escala="estimado",ax1=None, ax2=None):
    shape, loc, scale = weibull_min.fit(data, floc=0)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if escala!="estimado":
        scale = escala
        
    if forma!="estimado":
        shape = forma
    
    n = len(data)
    sm.qqplot(data,weibull_min,distargs=(shape,),loc=loc,scale=scale,line='45',ax=ax1)
    # Se agrega un título a la gráfica
    ax1.set_title("Q-Q Plot Weibull")
    
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(weibull_min.cdf(data,shape,loc=loc,scale=scale))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax2)
    ax2.set_title('P-P plot Weibull')
    ax2.set_xlabel('Theoretical Probabilities')
    ax2.set_ylabel('Sample Probabilities')
    ax2.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    ax2.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    return (shape, scale)

def ajuste_observada_teorica(**kwargs):
    dist= kwargs['dist']
    data=kwargs['data']
    tamanio_muestra = kwargs['tamanio_muestra']
    ax= kwargs['ax']

    if dist=='normal':
      media= kwargs['media']
      desvesta= kwargs['desvesta']

    if dist=='lognormal':
      media= kwargs['media']
      desvesta= kwargs['desvesta']

    if dist=='exponencial':
      tasa= kwargs['tasa']

    if dist=='uniforme':
      minimo= kwargs['minimo']
      maximo= kwargs['maximo']
    
    if dist=='triangular':
      minimo= kwargs['minimo']
      moda=kwargs['moda']
      maximo= kwargs['maximo']
    
    
    if dist=='gamma':
      media= kwargs['media']
      varianza= kwargs['varianza']
    
    if dist == "weibull":
      forma= kwargs['forma']
      escala= kwargs['escala']


    ax.hist(data, bins=int(math.sqrt(tamanio_muestra)), density=True, color="#3182bd", alpha=0.5, 
            label='Datos')
    ax.set_xlabel('Tiempo Arribos')
    ax.set_title('Distribucion ' + dist )

    x = np.linspace(min(data), max(data), tamanio_muestra)
    if dist=='normal':
      pdf = norm.pdf(x, loc=media, scale=desvesta)

    if dist=="lognormal":
      pdf = lognorm.pdf(x, s=desvesta, scale=np.exp(media))

    if dist=='exponencial':
      pdf = np.exp(-x / tasa) / tasa

    if dist == "uniforme":
      x = np.linspace(minimo , maximo , tamanio_muestra)
      pdf = np.piecewise(x, [x < minimo, (minimo <= x) & (x <= maximo), x > maximo],
                            [0, 1 / (maximo - minimo), 0])

    if dist == "triangular":
      pdf= triang.pdf(x, c=(moda - minimo) / (maximo -  minimo), loc=minimo, scale=maximo - minimo)
      
    if dist=='gamma':
      escala= (varianza /media)
      shape = (1/(varianza*escala))
      pdf = gamma.pdf(x, shape)  
    
    if dist == "weibull":
      pdf = weibull_min.pdf(x, forma)
  
    ax.plot(x, pdf , 'r', lw=2, label='FDP ' + dist)

    ax.legend()
