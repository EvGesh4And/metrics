import sys
sys.path.append('..')

from matplotlib import pyplot as plt

try:
    from .estimator_utils import time_mapper, Plotmatrix
except ImportError:
    from utils.estim.estimator_utils import time_mapper, Plotmatrix


def steptest_estimator(config, data, plot_time='minutes', vars=None, plot_raw=3):

    if type(vars)==type(None): vars=[]
    
    result = data.copy(deep=True)
    
    # наборы переменных для отрисовки
    MV_set = config['MV'].keys()
    CV_set = config['CV'].keys()
    DV_set = config['DV'].keys()

    # если надо отрисовать конкретные переменные, оставляем только их
    if vars:
        MV_set = [var for var in MV_set if var in vars]
        CV_set = [var for var in CV_set if var in vars]
        # DV_set = [var for var in DV_set if var in vars]


    pmatrix = Plotmatrix(MV_set, CV_set, DV_set, plot_raw) # объект с размерами матрицы графиков

    result.index = ((result.index - result.index.min()) \
                / time_mapper[plot_time]) # переводим время в нужные ед. и начинаем с 0

    fig_heith = 10 # высота графиков
    if pmatrix.width == 1: fig_heith = 15 # увеличиваем, если все графики в стоблце

    x_label = f'time, {plot_time}' # наименование временной шкалы    
    fig, axs = plt.subplots(pmatrix.length, pmatrix.width, figsize=(15,fig_heith)) # фигура с матрицей графиков

    # формирование графиков для каждой CV
    for CV in CV_set:

        # графики
        position = pmatrix.get_position()

        axs[position].plot(result.index, result[CV])
        axs[position].set_xlabel(x_label)
        axs[position].set_ylabel(CV)
        
        # обновляем положение для следующего графика кроме последнего эл-та
        if CV != list(CV_set)[-1]: pmatrix.update_position()

    pmatrix.lower_position() # строим графики с новой строки в матрице
  
    # формирование графиков для каждой MV и DV
    for vars in [MV_set, DV_set]:
        for var in vars:

            # графики
            position = pmatrix.get_position()

            axs[position].plot(result.index, result[var])
            axs[position].set_xlabel(x_label)
            axs[position].set_ylabel(var)

            # обновляем положение для следующего графика
            pmatrix.update_position()

    # отображаем графики
    fig.tight_layout()
    plt.show()