# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 00:18:03 2021

@author: arnau
"""
class Regr:
  def xs(self, valor, num_xs):
    import math
    return round(valor, (num_xs - int(math.floor(math.log10(abs(valor))))-1))
class Regr(Regr):
  def pend(self, a, b):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return self.xs(float(model.coef_),5)
  def i_pend(self, a, b):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    S_xx=np.std(x, ddof = 1)**2*(len(x)-1)
    S_yy=np.std(y, ddof = 1)**2*(len(y)-1)
    model = LinearRegression().fit(x, y)
    S_x_y=np.sqrt((S_yy-(S_xx*float(model.coef_)**2))/(len(x)-2))
    inc_pen = S_x_y/np.sqrt(S_xx)
    return self.xs(inc_pen, 3)
  def ordor(self, a, b):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    model = LinearRegression().fit(x, y)
    return self.xs(float(model.intercept_), 5)
  def i_ordor(self, a, b):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    S_xx=np.std(x, ddof = 1)**2*(len(x)-1)
    S_yy=np.std(y, ddof = 1)**2*(len(y)-1)
    model = LinearRegression().fit(x, y)
    S_x_y=np.sqrt((S_yy-(S_xx*float(model.coef_)**2))/(len(x)-2))
    x_sq=[]
    for i in x:
      x_sq.append(i**2)
    sq_x_mins_mean = []
    for i in x:
      sq_x_mins_mean.append((i-np.mean(x))**2)
    inc_or1 = S_x_y * np.sqrt((sum(x_sq))/(len(x)*(sum(sq_x_mins_mean))))
    inc_or2 = S_x_y * np.sqrt((sum(x_sq))/(len(x)*(len(x)-1)*np.std(x)**2))
    return self.xs(float(inc_or2), 3)
  def interpolation(self, a, b, signal, replicas):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    S_xx=np.std(x, ddof = 1)**2*(len(x)-1)
    S_yy=np.std(y, ddof = 1)**2*(len(y)-1)
    model = LinearRegression().fit(x, y)
    S_x_y=np.sqrt((S_yy-(S_xx*float(model.coef_)**2))/(len(x)-2))
    x_0 = (signal - float(model.intercept_))/float(model.coef_)
    s_0 = (S_x_y/float(model.coef_)) * np.sqrt(1/replicas + 1/len(y) + (signal-np.mean(y))**2/(float(model.coef_)**2 * S_xx))
    return [self.xs(float(x_0), 5), self.xs(float(s_0), 3)]
  def extrapolation(self, a, b):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    S_xx=np.std(x, ddof = 1)**2*(len(x)-1)
    S_yy=np.std(y, ddof = 1)**2*(len(y)-1)
    model = LinearRegression().fit(x, y)
    S_x_y=np.sqrt((S_yy-(S_xx*float(model.coef_)**2))/(len(x)-2))
    x_0 = abs(-float(model.intercept_)/float(model.coef_))
    s_0 = (S_x_y/float(model.coef_)) * np.sqrt(1/len(y) + np.mean(y)**2/(float(model.coef_)**2 * S_xx))
    return [self.xs(float(x_0), 5), self.xs(float(s_0), 3)]
  def cf_reg(self, a, b):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return self.xs(r_sq, 4)
  def graf(self, a, b, e_x, e_y, tit_a, tit_b):
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import numpy as np
    x = np.array(a).reshape((-1, 1))
    y = np.array(b)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    fig = plt.figure(figsize=(6.5,4))
    ax = fig.add_subplot(111)
    ax.set(xlabel=tit_a, ylabel=tit_b)
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,2), useOffset = False, useMathText=True) #or style plain
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(8)  #mida lletres eix y
    ax.plot(x,y_pred, color='coral', lw=2, linestyle=(0, (5, 4)), zorder=1)
    ax.scatter(x,y, color= 'teal', lw=0.3, zorder=3)
    for i in range(len(x)):
      ax.errorbar(x[i], y[i], xerr=e_x[i], yerr=e_y[i],capsize=3 ,color='k', alpha =0.8,zorder=2)
    plt.savefig(str(Regr().ordor(a, b))+'_'+str(Regr().pend(a, b))+'_'+str(Regr().cf_reg(a,b))+'_FigurePlot.png', dpi=1000)
    plt.show()
  def plot_eq(self, a, b):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.axis('off')
    plt.text(-.1, .9, r'Equation from linear optimization:')
    plt.text(-.1,.7, r'$y = a + bx$')
    plt.text(-.1,.5, r'$a=$'+str(Regr().ordor(a, b))+r'$\pm$'+str(Regr().i_ordor(a, b)))
    plt.text(-.1,.3, r'$b=$'+str(Regr().pend(a, b))+r'$\pm$'+str(Regr().i_pend(a, b)))
    plt.text(-.1,.1, r'$R^2=$'+str(Regr().cf_reg(a,b)))
    plt.savefig(str(Regr().ordor(a, b))+'_'+str(Regr().pend(a, b))+'_'+str(Regr().cf_reg(a,b))+'_eqPlot.pdf')
    plt.savefig(str(Regr().ordor(a, b))+'_'+str(Regr().pend(a, b))+'_'+str(Regr().cf_reg(a,b))+'_eqPlot.png', dpi=1000)

x=[1, 2, 3, 4]
y=[.086/.292, .177/.292, .259/.295, .35/.294]
errx=[0 for i in x]
erry=[0 for i in y]
signal = .254/.233
replicas = 1
tit_x = r"[Co] $\mu$g/mL"
tit_y = "Senyal del Co"
interpolation_results = Regr().interpolation(x, y, signal, replicas)
extrapolation_results = Regr().extrapolation(x, y) #only for standard addition
print('Pendent:',Regr().pend(x,y))
print('Ordenada a l\'origen:',Regr().ordor(x,y))
print('Coeficient de regressió:',Regr().cf_reg(x,y))
print('Incertesa de l\'ordenada a l\'origen:',Regr().i_ordor(x,y))
print('Incertesa del pendent:',Regr().i_pend(x,y))
print('Resultat interpolat:', interpolation_results[0])
print('Incertesa resultat interpolat:', interpolation_results[1])
print('Resultat extrapolat:', extrapolation_results[0])
print('Incertesa resultat extrapolat:', extrapolation_results[1])
Regr().graf(x, y, errx, erry, tit_x, tit_y)
Regr().plot_eq(x, y)