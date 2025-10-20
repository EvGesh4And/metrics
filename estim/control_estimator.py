import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from math import isnan
from statistics import mean
import copy

try:
    from .estimator_utils import time_mapper, Plotmatrix
except ImportError:
    from utils.estim.estimator_utils import time_mapper, Plotmatrix

def control_estimator(config, data, exp_vars=None, stability_time_error=3, plot_time='minutes', plot=True, to_png=False,
                      plot_raw=3, fig_height=5, fig_width=10, font_size=None, verbose_plot=True):
    
    if type(exp_vars)==type(None):
        exp_vars=[]

    target_postfix = "_target"  # добавление к имени CV - имя поля с таргетом
    min_border_postfix = "_min" # добавление к имени CV - имя поля с нижней границей
    max_border_postfix = "_max" # добавление к имени CV - имя поля с верхней границей

    # наборы переменных для отрисовки
    MV_set = config['data']['MV'].keys()
    CV_set = config['data']['CV'].keys()
    DV_set = config['data']['DV'].keys()    

    # Определение корректных границ целей    
    for CV in CV_set:
        target_name = CV + target_postfix # имя поля с таргетом
        min_name = CV + min_border_postfix # имя поля с прогнозом
        max_name = CV + max_border_postfix # имя поля с прогнозом

        data[min_name] = data[min_name].astype(float).apply(lambda x: -np.inf if np.isnan(x) else x)
        data[max_name] = data[max_name].astype(float).apply(lambda x: np.inf if np.isnan(x) else x)

        idx_none = data[(data[min_name] == -np.inf) & (data[max_name] == np.inf)].index
        data.loc[idx_none, min_name] = data.loc[idx_none, target_name].copy(deep=True).astype(dtype=np.float32)
        data.loc[idx_none, max_name] = data.loc[idx_none, target_name].copy(deep=True).astype(dtype=np.float32)
    
    # если надо отрисовать конкретные переменные, оставляем только их
    if exp_vars:
        MV_set = [var for var in MV_set if var in exp_vars]
        CV_set = [var for var in CV_set if var in exp_vars]
        DV_set = [var for var in DV_set if var in exp_vars]

    metrics = compute_metrics(config, data, MV_set, CV_set, DV_set, stability_time_error, plot_time)

    plot_metrics(config, data, metrics, MV_set, CV_set, DV_set, plot, plot_time, to_png,
                      plot_raw, fig_height, fig_width, font_size, verbose_plot)
    
    return metrics

def plot_metrics(config, data, metrics=None, MV_set=None, CV_set=None, DV_set=None, plot=True, plot_time='minutes', to_png=False,
                      plot_raw=3, fig_height=5, fig_width=10, font_size=None, verbose_plot=True):
    # принимает на вход конфиг и DataFrame результатом или callable-объект с аргументами
    # который будет вызван, чтоб получить DataFrame

    # Для сохраняемых графиков меняем шрифт
    if to_png != False and font_size is not None:
        plt.rcParams.update({'font.size': font_size})                

    if to_png: # убираем интерактивный режим, если сохраняем  графики в файл
        matplotlib.use('Agg')

    predict_postfix = "_predict"    # добавление к имени CV - имя поля с прогнозом модели, по рассчитанным MV
    ss_postfix = "_ss"              # добавление к имени CV - имя поля с ss модели
    min_border_postfix = "_min"     # добавление к имени CV - имя поля с нижней границей
    max_border_postfix = "_max"     # добавление к имени CV - имя поля с верхней границей

    result = data.copy(deep=True)

    # Функция для проверки "мусора" и округления чисел
    def clean_up_number(x, threshold=1e-6):
        if isinstance(x, (int, float)):  # Проверяем, является ли x числом
            rounded_value = round(x, 6)
            if abs(x - rounded_value) < threshold:
                return rounded_value
            else:
                return x
        return x  # Возвращаем без изменений, если это не число

    # Применяем очистку для всего DataFrame
    result = result.applymap(lambda x: clean_up_number(x))

    result['Time'] = result.index # добавляем колонку с временем
    result = result.reset_index(drop=True) # меняем индекс на нумерацию строк
   
    x_label = f'Time, {plot_time}' # наименование шкалы времени
    result['Time'] = result['Time'] / time_mapper[plot_time] # переводим время в нужные ед. и начинаем с 0
    
    pmatrix = Plotmatrix(MV_set, CV_set, DV_set, plot_raw) # объект с размерами матрицы графиков
    # if pmatrix.width == 1: fig_height = 15 # увеличиваем ширину, если все графики в стоблце
    if plot: 
        fig, axs = plt.subplots(pmatrix.length, pmatrix.width, figsize=(pmatrix.width*fig_width, pmatrix.length*fig_height)) # фигура с матрицей графиков

    # ф-я расчета конечной позиции графика
    def lower_position(df_col):
        delta = df_col.max() - df_col.min()
        return not df_col.min() <= df_col.iloc[-1] <= df_col.min() + delta / 2
    
    # расчет метрик и формирование графиков для каждой CV
    for CV in CV_set:

        predict_name = CV + predict_postfix
        ss_name = CV + ss_postfix
        min_name = CV + min_border_postfix 
        max_name = CV + max_border_postfix 
        
        mrbe_plot = round(metrics['CV'][CV]['mrbe'], 2)  # Округление для графиков
        emrbe_plot = round(metrics['CV'][CV]['emrbe'], 2)  # Округление для графиков

        stability_time_plot = round(metrics['CV'][CV]['stab_first'][0], 2)
        overcontrol_plot = round(metrics['CV'][CV]['overcontrol'][0], 2) # округление для графиков

        
        if plot: # графики
            position = pmatrix.get_position()            
            axs[position].plot(result['Time'], result[CV],
                               label=f"Тек. знач., eMRBE: {emrbe_plot}%")
            if verbose_plot:
                try:
                    axs[position].plot(result['Time'], result[predict_name], label=f"прогноз, MRBE: {mrbe_plot}%")
                except:
                    pass
            axs[position].plot(result['Time'], result[max_name], label=f"Верхн. пред., ovc: {overcontrol_plot}%")  

            if verbose_plot:
                axs[position].plot(result['Time'], result[min_name], label=f"Нижн. пред., stab_first: {stability_time_plot}")
            else:
                axs[position].plot(result['Time'], result[min_name], label=f"Нижн. пред.")

            axs[position].plot(result['Time'], np.where(result['optim'], result[ss_name], np.nan), label="Уст. знач. Опт. ВКЛ.", color="pink") 
            if verbose_plot:
                axs[position].plot(result['Time'], np.where(result['optim'], np.nan, result[ss_name]), label="Уст. знач. Опт. ВЫКЛ.", color="gray")           
            
            num_cv = list(config["data"]["CV"].keys()).index(CV)
            if result['optim'].any() and "x_k_quad" in config["controller"].keys():
                x_k_quad_val = config["controller"]["x_k_quad"][num_cv]
                if x_k_quad_val != 0  and x_k_quad_val is not None:
                    x_desired_val = config["controller"]["x_desired"][num_cv]
                    axs[position].plot(result['Time'], [x_desired_val] * result.shape[0], label=f"Цель")
            num_cv = num_cv + 1
            axs[position].set_xlabel(x_label)
            axs[position].set_ylabel(f'CV{num_cv}: {CV}')
            axs[position].legend(loc='lower right' if lower_position(result[CV]) else 'upper right')

            # обновляем положение для следующего графика кроме последнего эл-та
            if CV != list(CV_set)[-1]: pmatrix.update_position()

    pmatrix.lower_position() # строим графики с новой строки в матрице

    # расчет метрик и формирование графиков для каждой MV и DV
    for vars in [MV_set, DV_set]:

        vars_name = 'MV' if vars == MV_set else 'DV'

        for var in vars:

            plot_name = var

            if vars_name == 'MV':
                delta_mean_plot = round(metrics['MV'][var]['dmean'], 2) # округление для графиков
                plot_name = f"Тек. знач., d_mean: {delta_mean_plot}%" # добавляем метрику на график


            if plot: # графики
                position = pmatrix.get_position()
                axs[position].plot(result['Time'], result[var], label=plot_name)                
                axs[position].set_xlabel(x_label)

                # Верхний и нижний пределя для MV
                if vars_name == 'MV':
                    mv_max = [config['controller']['u_max'][list(config['data']['MV'].keys()).index(var)]] * result.shape[0]
                    mv_min = [config['controller']['u_min'][list(config['data']['MV'].keys()).index(var)]] * result.shape[0]                    
                    axs[position].plot(result['Time'], mv_max, label=f"Верхн. пред.")
                    axs[position].plot(result['Time'], mv_min, label=f"Нижн. пред.")                    

                # Желательное значение MV
                if vars_name == 'MV':
                    ss_name = var+ss_postfix
                    axs[position].plot(result['Time'], np.where(result['optim'], result[ss_name], np.nan), 
                                       label="Уст. знач. Опт. ВКЛ.", color="pink")
                    if verbose_plot:
                        axs[position].plot(result['Time'], np.where(result['optim'], np.nan, result[ss_name]), 
                                        label="Уст. знач. Опт. ВЫКЛ.", color="gray")
                    if result['optim'].any() and "u_k_quad" in config["controller"].keys():
                        num_mv = list(config["data"]["MV"].keys()).index(var)
                        u_k_quad_val = config["controller"]["u_k_quad"][num_mv]
                        if u_k_quad_val != 0. and u_k_quad_val is not None:
                            u_desired_val = config["controller"]["u_desired"][num_mv]
                            axs[position].plot(result['Time'], [u_desired_val] * result.shape[0], label=f"Цель")
                
                var_num = list(config['data'][vars_name].keys()).index(var) + 1
                axs[position].set_ylabel(f'{vars_name}{var_num}: {var}')
                axs[position].legend(loc='lower right' if lower_position(result[var]) else 'upper right')

                # обновляем положение для следующего графика
                pmatrix.update_position()
                fig.tight_layout()
    
    if plot and to_png:
        plt.savefig(f'{to_png}.png')

def compute_metrics(config, data, MV_set=None, CV_set=None, DV_set=None, stability_time_error=3, plot_time='minutes'):
    
        # ф-я, считающая время попадания в границы
    def achive_time_border(CV, elem):                

        # индекс, когда 1й раз попали в границы
        achieve_time_idx = ((result[CV].loc[elem['start_idx']:elem['end_idx']] >= elem['value_min']) \
                            & (result[CV].loc[elem['start_idx']:elem['end_idx']] <= elem['value_max'])).idxmax()
        achieve_time = result['Time'].loc[achieve_time_idx]
        # если idxmax вернет 0й индекс, это значит, никогда не достигли. Заменим на 'inf'
        if achieve_time_idx == elem['start_idx']: 
            achieve_time = 'inf'
        return achieve_time, achieve_time_idx
    
    # ф-я, считающая время достежения целей
    def achive_time_target(CV, elem):                

        # индекс, когда 1й раз достигли целей
        if result[CV].loc[elem['start_idx']] < elem['value_max']: # если цель была "выше"
            achieve_time_idx = (result[CV].loc[elem['start_idx']:elem['end_idx']] - elem['value_max'] >= 0).idxmax()
        else: # если "ниже"
            achieve_time_idx = (result[CV].loc[elem['start_idx']:elem['end_idx']] - elem['value_max'] <= 0).idxmax()
        achieve_time = result['Time'].loc[achieve_time_idx]
        # если idxmax вернет 0й индекс, это значит, никогда не достигли. Заменим на 'inf'
        if achieve_time_idx == elem['start_idx']: 
            achieve_time = 'inf'
        return achieve_time, achieve_time_idx

    # ф-я, считающая время стабилизации, относительно границ
    def stability_time_border(CV, elem):      
        
        middle = (elem['value_max'] + elem['value_min']) / 2 # середина диапазона
        # ряд, когда CV отклонялось более, чем на stability_time_error % с вычетом периода окна в конце
        error_exceeding = (((result[CV] - elem['value_max']) * (100 / middle) > stability_time_error) | \
        ((elem['value_min'] - result[CV]) * (100 / middle) > \
         stability_time_error)).loc[elem['start_idx']: elem['end_idx']]
        try: # если превышение было в последнем индексе и ошибка запроса индекс + 1,
             # или если получаем пустой Series - обходим ошибки. Итоговое время будет 'inf'
            stability_time_idx = error_exceeding[::-1].idxmax() # считаем с конца первое превышение
            stability_time = result['Time'].loc[stability_time_idx + 1]
        except:
            if stability_time_idx == elem['end_idx']: # обход ошибки индекс + 1 для последней ступеньки
                pass
            else: # обход прочих ошибок
                stability_time = 'inf'
                return stability_time, elem['end_idx']
        # если idxmax вернет последний индекс с вычетом окна
        if stability_time_idx == elem['end_idx']: 
            if error_exceeding.sum() == 0: # значит достигли сразу и ряд с ошибками будет полностью False
                stability_time = result['Time'].loc[elem['start_idx']]
            else: # или никогда не достигли
                stability_time = 'inf'
        return stability_time, stability_time_idx

    # ф-я рассчета величины максимального перерегулирования после попадания в диапазон (высота "горба" в %)
    def overcontrol_border(CV, achive_time_idx, elem, norm_coeff):

        # Перерегулирование вниз по нормированному коэффициенту
        overcontrol_min = ((result[CV] * (-1) + elem['value_min']) * (100 / norm_coeff)).loc[achive_time_idx:elem['end_idx']].max()
        
        # Перерегулирование вверх по нормированному коэффициенту
        overcontrol_max = ((result[CV] - elem['value_max']) * (100 / norm_coeff)).loc[achive_time_idx:elem['end_idx']].max()
        
        # Максимальное значение перерегулирования
        overcontrol = max(overcontrol_min, overcontrol_max, 0)

        return overcontrol

    # ф-я, считающая среднее изменение MV за шаг
    delta_mean_calc = lambda x: (abs(x.shift() - x))[1:].mean()

    min_border_postfix = "_min" # добавление к имени CV - имя поля с нижней границей
    max_border_postfix = "_max" # добавление к имени CV - имя поля с верхней границей

    result = data.copy(deep=True)

    result['Time'] = result.index # добавляем колонку с временем
    result = result.reset_index(drop=True) # меняем индекс на нумерацию строк

    # справочники для сбора метрик
    CV_metrics = {}
    MV_metrics = {}

    metrics = {
        'CV' : CV_metrics,
        'MV': MV_metrics,
    }   
   
    result['Time'] = result['Time'] / time_mapper[plot_time] # переводим время в нужные ед. и начинаем с 0

    # расчет метрик и формирование графиков для каждой CV
    CV_norm = config['model']['k_norm_x']

    for i, CV in enumerate(CV_set):

        metric_values = {} # куда запишем значения метрик по этой CV
        min_name = CV + min_border_postfix # имя поля с прогнозом
        max_name = CV + max_border_postfix # имя поля с прогнозом

        # Нормализованный интеграл абсолютного выхода
        model_N = int(config['model']['N'])
        model_win = model_N if result.shape[0] > model_N else 0  # Длина окна

        for N in [1, model_win]:
            bottom_err = result[min_name][N:] - result[CV][N:]
            upper_err = result[CV][N:] - result[max_name][N:]

            bottom_err[bottom_err < 0] = 0  # Учитываем только выходы ниже границы
            upper_err[upper_err < 0] = 0  # Учитываем только выходы выше границы

            # Вычисление интеграла выхода за границы
            norm_integral = (bottom_err / CV_norm[i]).sum() + (upper_err / CV_norm[i]).sum()
            norm_integral = (norm_integral / len(result[CV][N:])) * 100  # Приводим к процентам

            if N == 1:
                metric_values['mrbe'] = norm_integral  # Записываем рассчитанную метрику
            else:
                metric_values['emrbe'] = norm_integral  # Записываем рассчитанную метрику

        # определение списка из словарей с целями / границами и диапазона индексов, когда они были
        start_ind = 1 # индекс элемента с первой целью, для которой считаем время достижения
        borders_idx = []
        idxs = result[(result[min_name] != result[min_name].shift()) | (result[max_name] != result[max_name].shift())].index[1:]
        for ind in idxs: # индексы, где меняли цели
            border_idx = {'value_min': result[min_name][start_ind]} # записываем значения границ
            border_idx['value_max'] = result[max_name][start_ind]
            border_idx['start_idx'] = start_ind # начало периода с этими границами
            border_idx['end_idx'] = ind # конец периода с целью
            start_ind = ind # перезаписываем индекс начала для следующих границ
            borders_idx.append(border_idx)

        # добавляем индекс последних границ
        border_idx = {'value_min': result[min_name][start_ind]}
        border_idx['value_max'] = result[max_name][start_ind]
        border_idx['start_idx'] = start_ind
        border_idx['end_idx'] = len(result) - 1
        borders_idx.append(border_idx)

        # вычислем метрики для каждой цели и кдажем в списки
        achieve_times = []
        stability_times = []
        overcontrols = []
        
        for elem in borders_idx: # для каждой цели со своими индексами начала и конца
            # achieve_time
            if elem['value_max'] == elem['value_min']:
                achieve_time, achieve_time_idx = achive_time_target(CV, elem)
            else:
                achieve_time, achieve_time_idx = achive_time_border(CV, elem)
            # убираем прошедшее время перед новыми границами
            if achieve_time != 'inf' and start_ind != 1: 
                achieve_time = achieve_time - result['Time'].loc[elem['start_idx']]
            achieve_times.append(achieve_time)
            # stability_time
            stability_time, _ = stability_time_border(CV, elem)
            if stability_time != 'inf' and start_ind != 1:
                stability_time = stability_time - result['Time'].loc[elem['start_idx']]
            stability_times.append(float(stability_time))
            # overcontrol
            overcontrol = (overcontrol_border(CV, achieve_time_idx, elem, CV_norm[i]))
            overcontrols.append(overcontrol)

        # обновляем словарь рассчитанными метриками
        metric_values['stab_first'] = stability_times
        metric_values['overcontrol'] = overcontrols
        CV_metrics[CV] = metric_values


    # расчет метрик и формирование графиков для каждой MV и DV
    MV_norm = config['model']['k_norm_u']

    for vars in [MV_set, DV_set]:

        vars_name = 'MV' if vars == MV_set else 'DV'
        for i, var in enumerate(vars):
            if vars_name == 'MV':
                metric_values = {} # куда запишем значения метрик по этой MV
                # среднее изменение MV за шаг
                dmean = (delta_mean_calc(result[var])/MV_norm[i])*100
                metric_values['dmean'] = dmean # добавляем в справочник метрик
                MV_metrics[var] = metric_values
    
    # добавляем усредненные метрики по CV / MV
    stat={'mrbe':'CV_mrbe_mean', 'emrbe':'CV_emrbe_mean','overcontrol':'CV_overcontrol_mean','dmean':'MV_dmean'}
    
    for key in stat.keys():
        values = [v[key] for v in metrics['CV'].values() if key in v] + [v[key] for v in metrics['MV'].values() if key in v]
        
        if values:
            if type(values[0])==list: 
                list_ = [value[0] for value in values]
                metrics[stat[key]]= mean(list_)              
            else:
                values = [el for el in values if not isnan(el)]
                metrics[stat[key]] = mean(values)
 
     # добавляем максимальное время стабилизации по CV
    values = [v['stab_first'] for v in metrics['CV'].values() if 'stab_first' in v] 
    if values:
        if type(values[0])==list: 
            list_ = [value[0] for value in values]  
            metrics['CV_stab_first'] = 'inf' if 'inf' in list_ else max(list_)                        
        else:
            metrics['CV_stab_first'] = 'inf' if 'inf' in values else max(values)             

    # === objective: opt_obj / opt_obj_ref ===
    ctrl = config.get("controller", {})
    is_opt = bool(ctrl.get("is_optimization", False))

    # маски включенных переменных (если ключей нет — считаем, что все включены)
    z_on = ctrl.get("z_is_on") or [True] * len(CV_set)
    u_on = ctrl.get("u_is_on") or [True] * len(MV_set)

    x_k_lin  = ctrl.get("x_k_lin")  or []
    x_k_quad = ctrl.get("x_k_quad") or []
    x_des    = ctrl.get("x_desired") or []

    u_k_lin  = ctrl.get("u_k_lin")  or []
    u_k_quad = ctrl.get("u_k_quad") or []
    u_des    = ctrl.get("u_desired") or []

    def _safe(v, default=0.0):
        return default if v is None else v

    def objective_row(row):
        total = 0.0

        # CV (x)
        for i, cv in enumerate(CV_set):
            if i < len(z_on) and not z_on[i]:
                continue
            x = row[cv]
            klin  = _safe(x_k_lin[i]  if i < len(x_k_lin)  else 0.0)
            kquad = _safe(x_k_quad[i] if i < len(x_k_quad) else 0.0)
            total += klin * x
            if kquad != 0.0 and i < len(x_des) and x_des[i] is not None:
                des = x_des[i]
                total += kquad * (x - des) ** 2

        # MV (u)
        for j, mv in enumerate(MV_set):
            if j < len(u_on) and not u_on[j]:
                continue
            u = row[mv]
            klin  = _safe(u_k_lin[j]  if j < len(u_k_lin)  else 0.0)
            kquad = _safe(u_k_quad[j] if j < len(u_k_quad) else 0.0)
            total += klin * u
            if kquad != 0.0 and j < len(u_des) and u_des[j] is not None:
                des = u_des[j]
                total += kquad * (u - des) ** 2

        return total

    obj_series_ref = result.apply(objective_row, axis=1)

    # среднее с защитой от NaN/пустоты
    mean_obj = obj_series_ref.mean(skipna=True)
    if pd.isna(mean_obj):
        mean_obj = 0.0

    metrics['opt_obj_ref'] = float(mean_obj)           # всегда считаем
    metrics['opt_obj']     = float(mean_obj) if is_opt else 0.0  # по флагу

    return metrics

def _extend_result(result, metric_name='', exp_result=None):

    # рекурсивная ф-я, которая разворачивает иерархиеский справочник с метриками в плоский
    if type(exp_result)==type(None): exp_result={}
    for elem in result:
        if type(result[elem]) != dict:
            if metric_name in ['CV', 'MV', 'DV','']:
                exp_result[elem] = result[elem]
            else:
                exp_result[metric_name + '_' + elem] = result[elem]
        else:
            metric_name = elem
            _extend_result(result[elem], metric_name, exp_result)
    return exp_result
    
class MetricsCollector:

    def __init__(self):
        self.full = None # атрибут для сбора детальных метрик по экспериментам
        self.main = None # атрибут для сбора детальных метрик по экспериментам

    def add_result(self, result, exp_name):

        _full_result = _extend_result(result) # все метрики. Переводим в плоский справочник        
        if _full_result.get('calc_time_per_step')==None: _full_result['calc_time_per_step']=''
        if _full_result.get('max_calc_time_per_step')==None: _full_result['max_calc_time_per_step']=''       
        if _full_result.get('total_time_per_step')==None: _full_result['total_time_per_step']=''
        if _full_result.get('max_total_time_per_step')==None: _full_result['max_total_time_per_step']='' 

        main_result = {}

        main_result['total_time_per_step'] = _full_result['total_time_per_step'] # главные метрики
        main_result['max_total_time_per_step'] = _full_result['max_total_time_per_step'] # главные метрики

        # берем первые значения их метрик-списков, которые нужны, если цели менялись
        for key, value in _full_result.items():
            if type(value) == list:
                _full_result[key] = value[0]
        
        # добавляем усредненные метрики по CV / MV и максимальное время стабилизации по CV
        for metrix in ['CV_mrbe_mean', 'CV_emrbe_mean', 'CV_overcontrol_mean',
                       'MV_dmean','CV_stab_first', 'opt_obj', 'opt_obj_ref', 'obj_val']:
            main_result[metrix] = _full_result.get(metrix, '')

        main_result['calc_time_per_step'] = _full_result['calc_time_per_step'] # дополнительные метрики
        main_result['max_calc_time_per_step'] = _full_result['max_calc_time_per_step'] # гдополнительные метрики
        del _full_result['calc_time_per_step']

        # сортировка в детальных метриках
        full_ordered = {} 
        for metrix in ['mrbe', 'emrbe', 'overcontrol', 'dmean', 'stab_first',
                       'calc_time', 'opt_obj', 'opt_obj_ref', 'obj_val']:
            for key, value in _full_result.items():
                if metrix in key:
                    full_ordered[key] = value
        
        # добавляем главные метрики в объект с детальными
        full_result = copy.deepcopy(main_result)
        full_result.update(full_ordered)
               
        
        if not type(self.full) == type(pd.DataFrame()): # создаем DF, если ранее не было
            self.full = pd.DataFrame(columns=full_result.keys())
            self.main = pd.DataFrame(columns=main_result.keys())

        self.full.loc[exp_name] = full_result.values() # добавляем результат в DF
        self.main.loc[exp_name] = main_result.values()
