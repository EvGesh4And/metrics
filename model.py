import torch as tr 
import numpy as np
import pandas as pd
from copy import deepcopy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

try:
    from ..utils.functions.io import save_json, read_json
    from ..utils.estim import compute_forecast_ss, SSComputeParams, gains, coeff_plot
    from ..utils.functions.io import read_json
    from .predict import Predict
except ImportError:
    from utils.functions.io import save_json, read_json
    from utils.estim import compute_forecast_ss, SSComputeParams, gains, coeff_plot
    from utils.functions.io import read_json
    from model.predict import Predict

class Model(Predict):

    def save(self, path):
        """
        Сохраняет модель

        Parameters
        ----------
        path : str
            Путь хранения файла. Модель хранится в формате json        
        """
        config = self.get_config() # получаем конфиг в виде словаря
        del config['model']['N'] # удаляем лишние элекменты

        export_config = {} # словарь, который будем выгружать

        # экспортируем не нулевые параметры
        if config['version']:
            export_config['version'] = config['version']
        if config['data']: 
            export_config['data'] = config['data']
        if config['model']:
            export_config['model'] = {}
            for param, param_value in config['model'].items():
                if param in self.params_input.keys():
                    if self.params_input[param] != param_value:
                        param_value = self.params_input[param]
                if param_value != None:
                    export_config['model'][param] = param_value
        
        export_config = self.rec_valid_config_type(export_config)
        # сохранение
        save_json(export_config, path)
    
    def load(self, entity):
        """
        Загружает модель

        Parameters
        ----------
        path : str, dict
            Путь хранения файла в случае str,
            словарь конфига если dict
        """
        if entity:
            # Определяем вид переданной сущности
            if isinstance(entity, dict):
                config = entity
            elif isinstance(entity, str):
                config = read_json(entity)            

            # Получаем версию конфига
            self.get_config_version()
            
            # Обновляем конфиг
            self.update_config(config=config)
            if self.__class__.__name__ == 'TF':                
                if 'W' in vars(self):
                    if vars(self)['W']:
                        if 'version' in config:
                            if float(config['version']) < self.config_version:                                               
                                    self.design()
                                    self.update_config({'version': self.config_version})
                        else:
                            self.design()
                            self.update_config({'version': self.config_version})


    def gains(self, size=1000, show=True, save_to=None): # отобразить матрицу гейнов

        coeff_plot(self.data, self.H_tf, size=size, show=show, save_to=save_to)

    def impulse_coefficients(self, size=1000, show=True, save_to=None): # отобразить матрицу step_response коэффициентов
        
        coeff_plot(self.data, self.H, size=size, show=show, save_to=save_to)

    # ─────────────────── ТОЛЬКО ПРОГНОЗ (open-loop + rolling) ───────────────────
    def estimate_forecast(
        self,
        data: pd.DataFrame,
        N: int | str = "max",
        use_history: bool = False,
        rolling_sample_pct: float | None = None,
        anchor_filter_len: int = 1,
        max_rows: int | None = None,
        max_auto_N: int | None = 200,
    ):
        """Считает прогноз open-loop и rolling и возвращает :class:`ForecastResult`.

        **Что происходит под капотом**

        * Если ``max_rows`` задан и данных больше, чем ``max_rows``,
          автоматически берутся последние ``max_rows`` строк. Информация об
          усечении попадёт в ``ForecastResult.preprocessing`` и
          ``ForecastResult.notes``.
        * При ``N="max"`` горизонт не превышает ``max_auto_N``. Это защищает
          от слишком длинных вееров на больших задачах. Фактическое значение
          ``N`` можно посмотреть в ``result.N`` или в ``result.summary()``.
        * ``rolling_sample_pct`` управляет прореживанием стартов rolling.
          ``None`` включает автоматическую эвристику, которая учитывает длину
          ряда, количество CV/MV и выбранный горизонт ``N``. Значение ``100``
          означает расчёт веера на каждом такте.
          Фактические параметры прореживания попадают в
          ``result.preprocessing['rolling_sampling']`` и ``result.notes``.
        * ``anchor_filter_len`` задаёт длину окна усреднения факта, которое
          используется как якорь при шаге модели.

        **Как использовать результат**

        .. code-block:: python

            res = model.estimate_forecast(df)
            res.summary()                          # текстовая сводка
            metrics = res.metrics_dict()           # словарь с метриками
            res.plot_openloop(cv_list=["CV1"])    # график факт vs open-loop
            res.plot_rolling(fan_stride=200)       # веер rolling
            res.to_json(include_predictions=True)  # сериализация в json

        Параметры
        ---------
        data : pandas.DataFrame
            Исходные данные (колонки CV/MV должны совпадать с моделью).
        N : {'max', 'min', int}, по умолчанию ``'max'``
            Глобальный горизонт прогноза. Число > 0 — использовать как есть.
        use_history : bool, по умолчанию ``False``
            Управляет прогревом по факту (warm-up).
        rolling_sample_pct : float или None, по умолчанию ``None``
            Доля тактов (в процентах), на которых рассчитывается rolling.
            ``None`` включает автоматический подбор доли.
        anchor_filter_len : int, по умолчанию ``1``
            Длина окна сглаживания факта при обновлении состояния модели.
        max_rows : int или None, по умолчанию ``None``
            Максимальное число строк данных. ``None`` — использовать все.
        max_auto_N : int или None, по умолчанию ``200``
            Ограничение на автоматически выбранный ``N`` при ``N='max'``.
        """
        # имена переменных берём из модели
        self.get_var_cols()
        mv_cols = self.mv_cols
        cv_cols = self.cv_cols

        model = deepcopy(self)
        W = getattr(model, "W", None)
        N_all = getattr(model, "N_all", None)

        df = data.copy()
        preprocessing: dict[str, object] = {}
        notes: list[str] = []

        if max_rows is not None:
            max_rows_val = int(max_rows)
            if max_rows_val <= 0:
                raise ValueError("max_rows должно быть положительным числом или None.")
            if len(df) > max_rows_val:
                original_len = len(df)
                df = df.iloc[-max_rows_val:].copy()
                preprocessing["rows"] = {
                    "original": int(original_len),
                    "used": int(len(df)),
                }
                notes.append(
                    f"Данные усечены до последних {max_rows_val} строк (из {original_len})."
                )

        params = SSComputeParams(
            N=N,
            use_history=use_history,
            alpha_bias=1.0,
            dt=1.0,
            tau_is_steps=True,
            rolling_sample_pct=rolling_sample_pct,
            anchor_filter_len=anchor_filter_len,
            max_auto_N=max_auto_N,
        )

        res = compute_forecast_ss(
            W=W,
            N_all=N_all,
            df=df,
            mv_cols=mv_cols,
            cv_cols=cv_cols,
            p=params,
        )
        if notes:
            res.notes.extend(notes)
        if preprocessing:
            res.preprocessing.update(preprocessing)
        return res


    # ─────────────────── ТОЛЬКО ГЕЙНЫ ───────────────────
    def estimate_gains(self, data, mv_cols=None, cv_cols=None,
                       z0=None, u0=None, win: int | None = None,
                       min_step: int = 1,
                       plot: bool = False, plot_save: bool = False):
        """
        Считает только гейны и их метрики. Возвращает pd.DataFrame.
        """
        import pandas as pd

        if z0 is None: z0 = []
        if u0 is None: u0 = []
        if win is None: win = self.N

        self.get_var_cols()
        if not mv_cols: mv_cols = self.mv_cols
        if not cv_cols: cv_cols = self.cv_cols

        gm = gains(model=self, data=data, mv_cols=mv_cols, cv_cols=cv_cols,
                   win=win, z0=z0, u0=u0, min_step=min_step,
                   plot=plot, plot_save=plot_save)
        return pd.DataFrame(gm)

    
    def compare(self, model2, size=1000, show=True, save_to=None):
        """
        Функция построения графиков fir коэффициентов 2 моделей
        model2: model.FIR или model.TF
            модель с которой сравнивается текущая
        size: int
            размер графиков если их количество меньше 100
        show: bool
            Выводить или нет графики
        save_to: str
            Название файла сохранения графика. Если не задано то график не созраняется.

        """

        H_tf1 = self.H_tf
        H_tf2 = model2.H_tf 

        # наборы переменных для отрисовки
        n_mv = len(self.mv_cols)
        n_cv = len(self.cv_cols)   

        # Для маленьких графиков используем plotly
        if n_mv * n_cv <= 100:

            fig = make_subplots(rows=n_mv, cols=n_cv, subplot_titles=self.cv_cols) # формы для графиков              
            
            for idx_mv, mv_col in enumerate(self.mv_cols): # для каждой MV
                for idx_cv, cv_col in enumerate(self.cv_cols): # для каждой СV

                    fig.add_trace(go.Scatter(y=np.array(H_tf1[idx_cv][idx_mv], self.numpy_float_type), 
                                             name=f'h1_{cv_col}"/"{mv_col}', line=dict(color="steelblue")), row=idx_mv + 1, col=idx_cv + 1)
                    fig.add_trace(go.Scatter(y=np.array(H_tf2[idx_cv][idx_mv], self.numpy_float_type), 
                                             name=f'h1_{cv_col}"/"{mv_col}', line=dict(color="orange")), row=idx_mv + 1, col=idx_cv + 1)        

                fig.update_yaxes(title_text=mv_col, row=idx_mv + 1, col=1)

            fig.update_layout(
                            width=size,
                            height=size,
                            font=dict(size=10)
                        )

            # Показываем орафик если включена такая опция
            if show:
                fig.show()
            
            # Сохраняем график если задан путь
            if save_to is not None:
                fig.write_image(save_to)
        
        # Для больших графиков используем matplotlib
        else:

            fig, axs = plt.subplots(n_cv, n_mv, figsize=(100, 100))

            for idx_mv, mv_col in enumerate(self.mv_cols): # для каждой MV
                for idx_cv, cv_col in enumerate(self.cv_cols): # для каждой СV
                        
                        axs[idx_cv, idx_mv].plot(range(len(H_tf1[idx_cv][idx_mv])), H_tf1[idx_cv][idx_mv], label=f'h1_{cv_col}"/"{mv_col}', color="steelblue")
                        axs[idx_cv, idx_mv].plot(range(len(H_tf2[idx_cv][idx_mv])), H_tf2[idx_cv][idx_mv], label=f'h1_{cv_col}"/"{mv_col}', color="orange")
                        axs[idx_cv, idx_mv].set_ylabel(mv_col)
                        axs[idx_cv, idx_mv].set_title(cv_col)
            
            # Показываем орафик если включена такая опция
            if show:
                plt.show()

            # Сохраняем график если задан путь
            if save_to is not None:
                fig.savefig(save_to)


    def k_norm(self, data=None):

        if data is not None:
            # Получаем доступные CV и MV из данных
            partial_cv_cols = [col for col in data.columns if col in self.cv_cols]
            partial_mv_cols = [col for col in data.columns if col in self.mv_cols]
 
            x = data[partial_cv_cols]
            u = data[partial_mv_cols] # получаем данные по MV+DV и CV из датасета

            # Если входная матрица управляющей переменной в формате pandas.DataFrame преобразуем в torch.Tensor
            if type(u) == pd.DataFrame:
                u = tr.tensor(u.values, dtype=self.torch_float_type)
            
            # Если входная матрица контролируемой переменной в формате pandas.DataFrame преобразуем в torch.Tensor
            if type(x) == pd.DataFrame:
                x = tr.tensor(x.values, dtype=self.torch_float_type)
            
            # Если входная матрица управляющей переменной в формате pandas.DataFrame преобразуем в torch.Tensor
            if type(u) == np.ndarray:
                u = tr.tensor(u, dtype=self.torch_float_type)
            
            # Если входная матрица контролируемой переменной в формате pandas.DataFrame преобразуем в torch.Tensor
            if type(x) == np.ndarray:                
                x = tr.tensor(x, dtype=self.torch_float_type)
            
            self.k_norm_u = u.max(dim=0).values - u.min(dim=0).values
            self.k_norm_x = x.max(dim=0).values - x.min(dim=0).values
        else:
            pass
        
        return self.k_norm_u, self.k_norm_x