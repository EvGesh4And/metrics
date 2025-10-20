import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean
import copy
from sklearn.metrics import mean_absolute_error


def _gain_plot_design(fig):

    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', minor_ticks="outside", gridcolor='grey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='grey')
    fig.update_traces(hoverinfo="all", hovertemplate="MV: %{x}<br>CV: %{y}")

def coeff_plot(config, H, size=1000, show=True, save_to=None):  # отрисовка графиков гейнов по коэффициентам  

    # наборы переменных для отрисовки
    cv_cols = list(config['CV'].keys())
    mv_cols = list(config['DV'].keys()) + list(config['MV'].keys())

    n_mv = len(mv_cols)
    n_cv = len(cv_cols)   

    # Для маленьких графиков используем plotly
    if n_mv * n_cv <= 100:

        fig = make_subplots(rows=n_mv, cols=n_cv, subplot_titles=cv_cols) # формы для графиков              
        
        for idx_mv, mv_col in enumerate(mv_cols): # для каждой MV
            for idx_cv, cv_col in enumerate(cv_cols): # для каждой СV

                fig.add_trace(go.Scatter(y=np.array(H[idx_cv][idx_mv]), name=' ', line=dict(color="orange")), row=idx_mv + 1, col=idx_cv + 1)                 

            fig.update_yaxes(title_text=mv_col, row=idx_mv + 1, col=1)

        fig.update_layout(
                        width=size,
                        height=size,
                        font=dict(size=10)
                    )
        _gain_plot_design(fig)

        # Показываем орафик если включена такая опция
        if show:
            fig.show()
        
        # Сохраняем график если задан путь
        if save_to is not None:
            fig.write_image(save_to)
    
    # Для больших графиков используем matplotlib
    else:

        fig, axs = plt.subplots(n_cv, n_mv, figsize=(100, 100))

        for idx_mv, mv_col in enumerate(mv_cols): # для каждой MV
            for idx_cv, cv_col in enumerate(cv_cols): # для каждой СV
                    
                    axs[idx_cv, idx_mv].plot(range(len(H[idx_cv][idx_mv])), H[idx_cv][idx_mv], label=f'{cv_col}_{mv_col}', color="orange")
                    axs[idx_cv, idx_mv].set_ylabel(mv_col)
                    axs[idx_cv, idx_mv].set_title(cv_col)
        
        # Показываем орафик если включена такая опция
        if show:
            plt.show()

        # Сохраняем график если задан путь
        if save_to is not None:
            fig.savefig(save_to)

def _get_steps(data, min_len=20):  # min_len — минимальная длина шага
    steps = []
    start_idxs = list(data[(data != data.shift())].index[1:])
    end_idxs = list(data[(data != data.shift(-1))].index[1:])

    for ind_num, ind in enumerate(start_idxs): # для каждого индекса начала ступеньки определяем
        try:
            step = {
                # номер ступеньки
                'step_num': ind_num + 1,
                # предыдущее, текущее значение MV и разницу
                'prev_value': data[start_idxs[ind_num - 1]] if ind_num else data.iloc[0],
                'mv_value': data[ind],
                'delta': data[ind] - (data[start_idxs[ind_num - 1]] if ind_num else data.iloc[0]),
                # индексы начала, конца ступеньки и длительность в тактах (не в интервале времени)
                'start_idx': ind,
                'end_idx': end_idxs[ind_num],
                # предыдущие инндексы
                'prev_idx': end_idxs[ind_num - 1] if ind_num else data.index[0]
            }

            step_len = len(data.loc[step['start_idx']:step['end_idx']])
            if step_len < min_len:
                continue  # Пропускаем короткие шаги
            steps.append(step)
        except Exception as e:
            print(f"Ошибка при создании шага {ind_num}: {e}")

    return steps


def get_steps(data, mv_cols, win=1, min_step=0.25, min_len=20):
    _all_steps = {}
    for mv in mv_cols:
        _all_steps[mv] = _get_steps(data[mv], min_len=min_len)

    index = data.index.to_numpy()
    for mv, steps in _all_steps.items():
        all_other_steps = {k: v for k, v in _all_steps.items() if k != mv}
        for step_num, step in enumerate(steps):
            for other_mv, other_steps in all_other_steps.items():
                for other_step in other_steps:
                    if step['start_idx'] <= other_step['start_idx'] <= step['end_idx']:
                        pos = index.searchsorted(other_step['start_idx'] - 0.01, side='right') - 1
                        if pos >= 0:
                            _all_steps[mv][step_num]['end_idx'] = index[pos]

    all_steps = {}
    final_min_len = int(win * min_step)
    for mv, _steps in _all_steps.items():
        steps = []
        for step_num, step in enumerate(_steps):
            step_len = len(data.loc[step['start_idx']:step['end_idx']])
            step['step_len'] = step_len
            if step_len >= final_min_len:
                steps.append(step)
        all_steps[mv] = steps

    return all_steps

def gains(model, win, data, mv_cols=None, cv_cols=None, z0=None, u0=None, min_step=0.25, plot=True, plot_save=False):
    # отрисовка графиков гейнов и расчет метрик
    # на вход принимает датасет, наименования колонок CV и MV, функцию прогноза и окно

    # Список полей MV
    if mv_cols is None:
        mv_cols = list(model.get_config()['data']['MV']) + list(model.get_config()['data']['DV'])
    
    # Список полей CV
    if cv_cols is None:
        cv_cols = list(model.get_config()['data']['CV'])
    
    # Начальные значения CV
    if not isinstance(z0, np.ndarray):
        if z0 == None:
            z0 = model.get_config()['model']['z0']
    
    # Начальные значения MV
    if not isinstance(z0, np.ndarray):
        if u0 == None:
            u0 = model.get_config()['model']['z0']

    n_mv = len(mv_cols)
    n_cv = len(cv_cols)
    z0 = np.zeros(n_cv) # прогнозный гейн считаем из 0й точки
    u0 = np.zeros(n_mv)
    gain_metrics = []

    steps = get_steps(data=data, mv_cols=mv_cols, win=win, min_step=min_step)

    # Для маленьких наборов CV и MV
    if n_mv * n_cv <= 100:

        fig = make_subplots(rows=n_mv, cols=n_cv, # формы для графиков
                            subplot_titles=cv_cols)

        for idx_mv, mv_col in enumerate(mv_cols): # для каждой MV

            # создаем массив единичных значений MV 
            u = np.insert(np.zeros((win, n_mv - 1)), idx_mv, np.ones(win), axis=1)
            # считаем гейн-предикт на 1 ступеньку
            gain_pred = model.predict(z0=z0, u0=u0, u=u)
            gain_pred = np.insert(gain_pred, 0, z0, axis=0).T # объединяем с нулевой точкой
            gain_pred = pd.DataFrame(gain_pred.T, columns=cv_cols)

            for idx_cv, cv_col in enumerate(cv_cols): # для каждой СV

                # справочник для метрик
                metrics = {'cv': cv_col}
                metrics['mv'] = mv_col
                
                # строим график с фактами, получаем метрики
                metrics.update(estimate_gain(u=data[mv_col], z=data[cv_col], steps=steps[mv_col], model_gain=gain_pred[cv_col],
                                        fig=fig, fig_row=idx_mv + 1, fig_col=idx_cv + 1))

                gain_metrics.append(metrics) # добавляем в общий список метрик

                # строим график с прогнозным гейном модели
                fig.add_trace(go.Scatter(y=gain_pred[cv_col], 
                                        name='predict', line=dict(color="orange", width=3)), row=idx_mv + 1, col=idx_cv + 1)
                            

            fig.update_yaxes(title_text=mv_col, row=idx_mv + 1, col=1)
        fig.update_layout(
                        title = "Gain-ы модели и фактов",
                        width=1000,
                        height=1000,
                        font=dict(size=10)
                    )
        _gain_plot_design(fig)
        if plot: fig.show()
        if plot_save: fig.write_image(plot_save)
            
    
    # Для больших наборов CV и MV
    else:
        fig, axs = plt.subplots(n_mv, n_cv, figsize=(100, 100), squeeze=False)

        for idx_mv, mv_col in enumerate(mv_cols): # для каждой MV

            # создаем массив единичных значений MV 
            u = np.insert(np.zeros((win, n_mv - 1)), idx_mv, np.ones(win), axis=1)
            # считаем гейн-предикт на 1 ступеньку
            gain_pred = model.predict(z0=z0, u0=u0, u=u)
            gain_pred = np.insert(gain_pred, 0, z0, axis=0).T # объединяем с нулевой точкой
            gain_pred = pd.DataFrame(gain_pred.T, columns=cv_cols)

            for idx_cv, cv_col in enumerate(cv_cols): # для каждой СV

                # справочник для метрик
                metrics = {'cv': cv_col}
                metrics['mv'] = mv_col

                # строим график с фактами, получаем метрики
                metrics.update(estimate_gain(u=data[mv_col], z=data[cv_col], steps=steps[mv_col], model_gain=gain_pred[cv_col],
                                             fig=fig, axs=axs, fig_row=idx_mv + 1, fig_col=idx_cv + 1, plotly=False))
                
                gain_metrics.append(metrics) # добавляем в общий список метрик

                # строим график с прогнозным гейном модели
                axs[idx_mv, idx_cv].plot(gain_pred[cv_col], label='predict', color="orange")

        if plot: fig.show()
        if plot_save: fig.savefig(plot_save)


    return gain_metrics #gain_value, gain_mae, cover

def estimate_gain(u=None, z=None, steps=None, model_gain=None, fig=None, axs=None, fig_row=None, fig_col=None, plotly=True):
    # Проверяем, были ли переданы данные для u и z, если нет - инициализируем пустыми списками
    if u is None: u = []
    if z is None: z = []

    # Получаем словарь откликов на шаги, вызвав функцию collect_step_responses
    responses = collect_step_responses(steps, z)

    # Рисуем графики откликов на шаги с помощью функции plot_responses
    plot_responses(responses, steps, u, z, model_gain, fig, axs, fig_row, fig_col, plotly)

    # Рассчитываем метрики, используя данные откликов и модельный гейн
    metrics = calculate_metrics(model_gain, responses, steps)

    return metrics


def collect_step_responses(steps, z):
    """
    Собирает отклики на шаги для каждого шага и сохраняет их в словарь.
    Отклик на шаги - это изменение в значении z по отношению к старту шага (было - предыдущему значению).
    """
    responses = {}

    for step in steps:
        # Извлекаем параметры текущего шага
        step_num = step['step_num']
        delta = step['delta']
        start_idx = step['start_idx']
        end_idx = step['end_idx']

        # Вычисляем отклик по данным z
        y_fact = (z.loc[start_idx:end_idx] - z.loc[step['start_idx']])
        y_fact = pd.concat([pd.Series([0]), y_fact]) / delta  # Нормируем отклик на величину шага

        # Сохраняем отклик в словарь
        responses[step_num] = y_fact

    return responses


def calculate_metrics(model_gain, responses, steps):
    """
    Рассчитывает метрики для оценки качества модели.
    Включает MAE (среднюю абсолютную ошибку) и покрытие (разницу между прогнозируемым и фактическим значением).
    """
    mae = []  # Список для хранения значений MAE
    cover = []  # Список для хранения значений покрытия

    for _, step in enumerate(steps):
        # Извлекаем параметры текущего шага
        step_len = step['step_len']
        y_fact = responses[step['step_num']]

        # Длина окна гейна
        win = len(model_gain - 1)

        # Укорачиваем отклик, если он длиннее окна
        y_fact = y_fact.iloc[:win]

        # Вычисляем MAE для текущего шага
        mae.append(mean_absolute_error(model_gain.iloc[1:step_len + 1], y_fact.iloc[1:]))
        
        # Вычисляем покрытие
        cover.append(abs(model_gain.iloc[:step_len + 1].iloc[-1] - y_fact.iloc[-1]))

    # Усредняем метрики по всем шагам
    mae = mean(mae) if mae else 0
    cover = mean(cover) if cover else 0

    # Возвращаем метрики в виде словаря
    metrics = {'gain_value': model_gain.iloc[-1], 'mae': mae, 'cover': cover}

    return metrics


def plot_responses(responses, steps, u, z, model_gain, fig, axs, fig_row, fig_col, plotly):
    """
    Строит графики откликов на шаги.
    Использует Plotly или Matplotlib в зависимости от параметра plotly.
    """
    for step in steps:
        # Извлекаем параметры текущего шага
        step_num = step['step_num']
        delta = step['delta']
        start_idx = step['start_idx']
        end_idx = step['end_idx']

        # Получаем отклик для текущего шага
        y_fact = responses[step_num]

        # Длина окна гейна
        win = len(model_gain - 1)

        # Укорачиваем отклик, если он длиннее окна
        y_fact = y_fact.iloc[:win]
        
        # Формируем название для графика с описанием изменения
        plot_name = f'step{step_num}: {u.loc[start_idx]}->{u.loc[start_idx] + delta}<br>step{step_num}: {z.loc[start_idx]}->{z.loc[end_idx]}'

        # Строим график, используя Plotly или Matplotlib
        if plotly:
            fig.add_trace(go.Scatter(y=y_fact, name=plot_name, line=dict(color="steelblue", width=1)), row=fig_row, col=fig_col)
        else:
            axs[fig_row-1, fig_col-1].plot(range(len(y_fact)), y_fact, label=plot_name, color='steelblue')