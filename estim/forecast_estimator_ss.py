from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Union, Any, Sequence

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from .predict_ss import PredictSSRolling
except ImportError:
    from utils.estim.predict_ss import PredictSSRolling


# ───────────────────────── helpers ─────────────────────────

def _r2_percent(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    sse = np.sum((y - yhat) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return 0.0 if sst <= 0 else (100.0 * (1.0 - sse / sst))

def _smape(y, yhat, eps=1e-9):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return 100.0 * np.mean(2.0 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + eps))

def _compute_smape_r2(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan, np.nan
    return _smape(y_true[mask], y_pred[mask]), _r2_percent(y_true[mask], y_pred[mask])

def _z_ma_inclusive(Z: np.ndarray, t: int, win: int) -> np.ndarray:
    """
    Скользящее среднее факта по окну длиной win, включающее текущий индекс t.
    Если данных меньше win (в начале ряда) — усредняем по доступным.
    Z: shape (T, n_cv), возвращает (n_cv,).
    """
    win = int(max(1, win))
    t0 = max(0, t - win + 1)
    return Z[t0:t+1].mean(axis=0)


# ───────────────────────── результат расчёта ─────────────────────────

@dataclass
class ForecastResult:
    """Результат расчёта прогноза ``Model.estimate_forecast``.

    Экземпляр содержит исходные данные расчёта, подготовленные
    предсказания и агрегированные метрики. Объект можно использовать
    программно: сериализовать в ``dict``/``json`` и строить графики без
    повторного запуска расчёта. Основные сценарии использования:

    * ``result.summary()`` — компактный отчёт с ключевыми цифрами;
    * ``result.metrics_dict()`` или ``result.to_json()`` — метрики в
      машинном формате (например, чтобы сохранить в артефакты тестов);
    * ``result.plot_openloop(...)`` и ``result.plot_rolling(...)`` —
      интерактивные графики fact vs. forecast.
    """
    # ────────────── данные расчёта ──────────────
    df: pd.DataFrame
    mv_cols: Iterable[str]
    cv_cols: Iterable[str]

    # ────────────── результаты open-loop ──────────────
    openloop_pred: pd.DataFrame
    openloop_metrics_overall: pd.DataFrame
    openloop_R2_mean: float

    # ────────────── результаты rolling ──────────────
    rolling_pred_by_horizon: Dict[int, pd.DataFrame]
    rolling_metrics_per_horizon: pd.DataFrame
    rolling_metrics_overall: pd.DataFrame
    rolling_R2_mean: float

    # ────────────── служебная информация ──────────────
    N: int
    warmup_end_pos: int
    rolling_available: bool
    rolling_unavailable_reason: Optional[str] = None

    # ────────────── дополнительные сведения ──────────────
    notes: list[str] = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)

    # ----------------- сериализация -----------------
    def metrics_dict(self) -> Dict[str, Any]:
        """Возвращает словарь с основными метриками и служебной информацией.

        Используйте метод, если нужно сравнивать разные версии модели или
        зафиксировать метрики в юнит-/интеграционных тестах. Структура
        словаря полностью JSON-совместима: ``json.dumps(result.metrics_dict())``
        работает «из коробки».
        """

        def _df_to_index_dict(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
            if df is None or df.empty:
                return {}
            return df.to_dict(orient="index")

        def _df_to_hier_dict(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
            if df is None or df.empty:
                return {}
            return {
                str(idx): row for idx, row in df.to_dict(orient="index").items()
            }

        def _nan_to_none(value: Any) -> Any:
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                return None
            return value

        metrics: Dict[str, Any] = {
            "N": int(self.N),
            "warmup_end_pos": int(self.warmup_end_pos),
            "rolling_available": bool(self.rolling_available),
            "rolling_unavailable_reason": self.rolling_unavailable_reason,
            "notes": list(self.notes),
            "preprocessing": dict(self.preprocessing),
            "openloop": {
                "R2_mean": _nan_to_none(self.openloop_R2_mean),
                "metrics_by_cv": _df_to_index_dict(self.openloop_metrics_overall),
            },
            "rolling": {
                "R2_mean": _nan_to_none(self.rolling_R2_mean),
                "metrics_by_cv": _df_to_index_dict(self.rolling_metrics_overall),
                "metrics_per_horizon": _df_to_hier_dict(
                    self.rolling_metrics_per_horizon
                ),
            },
        }
        return metrics

    def to_dict(
        self,
        include_predictions: bool = False,
        include_source: bool = False,
        predictions_orient: str = "records",
    ) -> Dict[str, Any]:
        """Преобразует результат в словарь. Подходит для сериализации в JSON.

        По умолчанию возвращает только метрики. Передайте ``include_source``
        и/или ``include_predictions``, чтобы дополнительно получить исходные
        данные и предсказания (удобно для экспорта в отчёты или проверки
        регрессий).
        """

        result: Dict[str, Any] = {
            "metrics": self.metrics_dict(),
        }

        if include_source:
            result["data"] = self.df.reset_index().to_dict(orient=predictions_orient)

        if include_predictions:
            result["openloop_pred"] = self.openloop_pred.reset_index().to_dict(
                orient=predictions_orient
            )
            result["rolling_pred_by_horizon"] = {
                int(k): df.reset_index().to_dict(orient=predictions_orient)
                for k, df in self.rolling_pred_by_horizon.items()
            }

        return result

    def to_json(
        self,
        include_predictions: bool = False,
        include_source: bool = False,
        predictions_orient: str = "records",
        **json_kwargs: Any,
    ) -> str:
        """Возвращает строку JSON с результатами расчёта.

        Параметры ``include_predictions`` и ``include_source`` повторяют
        аргументы :meth:`to_dict`. Дополнительные ``json_kwargs`` пробрасываются
        в ``json.dumps`` (например, ``indent=2``).
        """

        payload = self.to_dict(
            include_predictions=include_predictions,
            include_source=include_source,
            predictions_orient=predictions_orient,
        )
        return json.dumps(payload, ensure_ascii=False, **json_kwargs)

    # ----------------- ФАКТ vs OPEN-LOOP -----------------
    def plot_openloop(
        self,
        start_pos: int | None = None,
        end_pos: int | None = None,
        title_prefix: str = "Open-loop",
        cv_list: Sequence[str] | None = None,
        mv_list: Sequence[str] | None = None,
        max_points: int | None = 2_000,
        show_messages: bool = True,
    ):
        """Факт vs open-loop на выбранном диапазоне.

        Параметры ``start_pos``/``end_pos`` задают полуинтервал
        ``[start_pos : end_pos)`` (правый конец не включается). Если их
        не передавать, по умолчанию берутся последние ``max_points`` точек —
        так график остаётся «лёгким» даже на больших кейсах. На графике
        автоматически показывается зона прогрева и метрики (sMAPE/R²) для
        каждой CV.

        Чтобы выбрать конкретные переменные:

        * ``cv_list`` — список колонок CV для сравнения факт/прогноз
          (по умолчанию первые 5);
        * ``mv_list`` — список MV, отображаемых на отдельном графике
          (по умолчанию первые 5).

        Если переданных переменных нет в наборе данных, будет выброшено
        ``ValueError`` — это помогает быстро обнаружить опечатку.
        """
        n = len(self.df)
        if start_pos is None and end_pos is None and max_points is not None and n > max_points:
            e = n
            s = max(0, n - int(max_points))
            auto_range = True
        else:
            s = 0 if start_pos is None else int(start_pos)
            e = n if end_pos   is None else int(end_pos)
            auto_range = False
        if not (0 <= s < n) or not (0 < e <= n) or not (s < e):
            raise ValueError(f"Неверный диапазон: start_pos={s}, end_pos={e}, len={n}. "
                             "Должно быть: 0 <= s < e <= len(df).")

        cv_all = list(self.cv_cols)
        if cv_list is None:
            cv_to_plot = cv_all[:5]
            auto_cv = True
        else:
            cv_to_plot = [str(cv) for cv in cv_list]
            missing = [cv for cv in cv_to_plot if cv not in cv_all]
            if missing:
                raise ValueError(f"Следующие CV отсутствуют в данных: {missing}")
            auto_cv = False
        if not cv_to_plot:
            raise ValueError("Нет CV для отображения (cv_list пуст).")

        mv_all = list(self.mv_cols)
        if mv_list is None:
            mv_to_plot = mv_all[:5]
            auto_mv = True
        else:
            mv_to_plot = [str(mv) for mv in mv_list]
            missing_mv = [mv for mv in mv_to_plot if mv not in mv_all]
            if missing_mv:
                raise ValueError(f"Следующие MV отсутствуют в данных: {missing_mv}")
            auto_mv = False

        dfw = self.df.iloc[s:e]
        zhw = self.openloop_pred.loc[dfw.index]
        L = len(dfw)
        x = np.arange(L)

        messages: list[str] = []
        if auto_range and show_messages:
            messages.append(
                f"Диапазон графика усечён до последних {L} точек. "
                "Передайте start_pos/end_pos для просмотра другого участка."
            )
        if auto_cv and len(cv_all) > len(cv_to_plot) and show_messages:
            messages.append(
                "Показаны только первые 5 CV. Передайте cv_list, чтобы указать другой набор."
            )
        if auto_mv and len(mv_all) > len(mv_to_plot) and show_messages and mv_to_plot:
            messages.append(
                "Показаны только первые 5 MV. Передайте mv_list, чтобы указать другой набор."
            )
        if messages:
            print("\n".join(messages))

        warmup_global = int(self.warmup_end_pos)
        warm_len_local = max(0, min(warmup_global, e) - s)

        for cv in cv_to_plot:
            y_pred = zhw[cv].astype(float).copy()
            if warm_len_local > 0:
                y_pred.iloc[:warm_len_local] = np.nan

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=dfw[cv], name="факт", mode="lines",
                                     line=dict(width=2)))

            if warm_len_local > 0:
                fig.add_vrect(x0=0, x1=warm_len_local - 1,
                              fillcolor="#9e9e9e", opacity=0.35, line_width=0, layer="below")
                fig.add_vline(x=warm_len_local - 0.5,
                              line_color="#616161", line_width=2, line_dash="dash")

            fig.add_trace(go.Scatter(x=x, y=y_pred, name="open-loop", mode="lines",
                                     connectgaps=False, line=dict(width=3)))

            if cv in getattr(self, "openloop_metrics_overall", pd.DataFrame()).index:
                sm = self.openloop_metrics_overall.loc[cv, "sMAPE_%"]
                r2 = self.openloop_metrics_overall.loc[cv, "R2_%"]
                fig.add_annotation(
                    xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                    text=f"sMAPE: {sm:.1f}%<br>R²: {r2:.1f}%", showarrow=False,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1
                )

            fig.update_layout(
                title=f"{title_prefix}: {cv}  |  [{s}:{e})",
                xaxis_title="Такт", yaxis_title=cv,
                width=1100, height=420,
                legend=dict(orientation="h", x=1, xanchor="right"),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.show()

        if mv_to_plot:
            fig_mv = go.Figure()
            for mv in mv_to_plot:
                fig_mv.add_trace(
                    go.Scatter(x=x, y=dfw[mv], name=mv, mode="lines", line=dict(width=2))
                )
            fig_mv.update_layout(
                title=f"{title_prefix}: MV  |  [{s}:{e})",
                xaxis_title="Такт",
                yaxis_title="MV",
                width=1100,
                height=420,
                legend=dict(orientation="h", x=1, xanchor="right"),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_mv.show()

    # ---------------------- «ВЕЕР» ROLLING ----------------------
    def plot_rolling(
        self,
        start_pos: int | None = None,
        end_pos: int | None = None,
        fan_stride: int = 100,
        fan_max: int = 20,
        title_prefix: str = "Rolling",
        cv_list: Sequence[str] | None = None,
        mv_list: Sequence[str] | None = None,
        max_points: int | None = 2_000,
        show_messages: bool = True,
    ):
        """Строит rolling «веер» на выбранном диапазоне.

        Диапазон ``[start_pos : end_pos)`` обрабатывается аналогично
        :meth:`plot_openloop`. По умолчанию берутся последние ``max_points``
        наблюдений и максимум 5 CV/MV. Аргументы ``fan_stride`` и ``fan_max``
        управляют прореживанием стартов rolling: например, ``fan_stride=200``
        и ``fan_max=10`` оставят каждый 200‑й старт, но не больше 10 штук.

        ``mv_list`` позволяет дополнительно отрисовать интересующие MV на
        отдельном графике (по умолчанию берутся первые 5), что помогает
        сопоставить динамику управлений и фактических CV.

        Если в выбранном диапазоне стартов недостаточно (учитывая прогрев и
        горизонт ``N``), метод бросает ``ValueError`` с подсказкой. Это нормальное
        поведение для коротких окон данных.
        """
        if not self.rolling_available:
            reason = self.rolling_unavailable_reason or "rolling не рассчитан."
            raise ValueError(f"Нельзя построить веер: {reason}")

        base_df = self.rolling_pred_by_horizon.get(1)
        if base_df is None or base_df.empty:
            raise ValueError("Нельзя построить веер: нет предсказаний для горизонта k=1.")

        n = len(self.df)
        if start_pos is None and end_pos is None and max_points is not None and n > max_points:
            e = n
            s = max(0, n - int(max_points))
            auto_range = True
        else:
            s = 0 if start_pos is None else int(start_pos)
            e = n if end_pos   is None else int(end_pos)
            auto_range = False
        if not (0 <= s < n) or not (0 < e <= n) or not (s < e):
            raise ValueError(f"Неверный диапазон: start_pos={s}, end_pos={e}, len={n}. "
                             "Должно быть: 0 <= s < e <= len(df).")

        cv_all = list(self.cv_cols)
        if cv_list is None:
            cv_to_plot = cv_all[:5]
            auto_cv = True
        else:
            cv_to_plot = [str(cv) for cv in cv_list]
            missing = [cv for cv in cv_to_plot if cv not in cv_all]
            if missing:
                raise ValueError(f"Следующие CV отсутствуют в данных: {missing}")
            auto_cv = False
        if not cv_to_plot:
            raise ValueError("Нет CV для отображения (cv_list пуст).")

        mv_all = list(self.mv_cols)
        if mv_list is None:
            mv_to_plot = mv_all[:5]
            auto_mv = True
        else:
            mv_to_plot = [str(mv) for mv in mv_list]
            missing_mv = [mv for mv in mv_to_plot if mv not in mv_all]
            if missing_mv:
                raise ValueError(f"Следующие MV отсутствуют в данных: {missing_mv}")
            auto_mv = False

        dfw = self.df.iloc[s:e]
        L = len(dfw)
        x = np.arange(L)

        messages: list[str] = []
        if auto_range and show_messages:
            messages.append(
                f"Диапазон графика усечён до последних {L} точек. "
                "Передайте start_pos/end_pos для просмотра другого участка."
            )
        if auto_cv and len(cv_all) > len(cv_to_plot) and show_messages:
            messages.append(
                "Показаны только первые 5 CV. Передайте cv_list, чтобы указать другой набор."
            )
        if auto_mv and len(mv_all) > len(mv_to_plot) and show_messages and mv_to_plot:
            messages.append(
                "Показаны только первые 5 MV. Передайте mv_list, чтобы указать другой набор."
            )
        if messages:
            print("\n".join(messages))

        warmup_global = int(self.warmup_end_pos)
        warm_len_local = max(0, min(warmup_global, e) - s)

        # валидные старты (внутри окна) с учётом горизонта
        valid_starts: list[int] = []
        for ts in base_df.index:
            try:
                t_full = self.df.index.get_loc(ts)
            except KeyError:
                continue
            t_pos = t_full - s
            if 0 <= t_pos < L and (t_pos + self.N) < L:
                valid_starts.append(t_pos)

        if warm_len_local > 0:
            valid_starts = [p for p in valid_starts if p >= warm_len_local]

        if not valid_starts:
            raise ValueError("В выбранном диапазоне нет валидных стартов для веера (учтите прогрев и горизонт N).")

        step = max(1, int(fan_stride))
        t_starts = valid_starts[::step]
        if len(t_starts) > fan_max:
            t_starts = t_starts[:fan_max]

        for cv in cv_to_plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=dfw[cv], name="факт", mode="lines",
                                     line=dict(width=2)))

            if warm_len_local > 0:
                fig.add_vrect(x0=0, x1=warm_len_local - 1,
                              fillcolor="#9e9e9e", opacity=0.35, line_width=0, layer="below")
                fig.add_vline(x=warm_len_local - 0.5,
                              line_color="#616161", line_width=2, line_dash="dash")

            for t0 in t_starts:
                ts = dfw.index[t0]
                fan_vals = []
                for k in range(1, self.N + 1):
                    pdk = self.rolling_pred_by_horizon.get(k)
                    if pdk is None or ts not in pdk.index:
                        break
                    fan_vals.append(pdk.loc[ts, cv])
                if len(fan_vals) < 2:
                    continue
                xf = np.arange(t0 + 1, t0 + 1 + len(fan_vals))
                fig.add_trace(go.Scatter(
                    x=xf, y=np.asarray(fan_vals), name=f"t={t0}", mode="lines",
                    line=dict(width=3), opacity=0.9
                ))

            fig.update_layout(
                title=f"{title_prefix}: «веер» (cv={cv})  |  [{s}:{e})",
                xaxis_title="Такт", yaxis_title=cv,
                width=1100, height=500,
                legend=dict(orientation="h", x=1, xanchor="right"),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.show()

        if mv_to_plot:
            fig_mv = go.Figure()
            for mv in mv_to_plot:
                fig_mv.add_trace(
                    go.Scatter(x=x, y=dfw[mv], name=mv, mode="lines", line=dict(width=2))
                )
            fig_mv.update_layout(
                title=f"{title_prefix}: MV  |  [{s}:{e})",
                xaxis_title="Такт",
                yaxis_title="MV",
                width=1100,
                height=420,
                legend=dict(orientation="h", x=1, xanchor="right"),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_mv.show()

    def summary(self, digits: int = 1, top_n: int = 3, return_text: bool = False) -> str | None:
        """Текстовая сводка расчёта.

        Метод помогает быстро оценить качество модели, не открывая графики.
        В отчёте показываются:

        * объём выборки и структура данных (сколько CV/MV);
        * выбранный горизонт ``N`` и позиция окончания прогрева;
        * заметки/примечания (например, об усечении датасета или капе ``N``);
        * агрегированные метрики open-loop и rolling с топ- и худшими CV;
        * при наличии — усреднённые по горизонту sMAPE/R².

        По умолчанию результат печатается. Если нужно получить строку (например,
        для логов или тестов), передайте ``return_text=True``.
        """
        import numpy as np
        import pandas as pd

        def fmt_df(df: pd.DataFrame) -> str:
            if df is None or df.empty:
                return "(нет данных)"
            dfr = df.copy()
            for col in dfr.columns:
                if np.issubdtype(dfr[col].dtype, np.number):
                    dfr[col] = dfr[col].astype(float).round(digits)
            return dfr.to_string()

        def pick_r2_column(df: pd.DataFrame) -> str | None:
            if df is None or df.empty:
                return None
            if "R2_%" in df.columns:
                return "R2_%"
            for c in df.columns:
                cn = str(c).lower()
                if "r2" in cn or "r²" in cn:
                    return c
            return None

        lines = []
        lines.append("=== Сводка прогноза ===")
        T = len(self.df)
        lines.append(f"Объём выборки (T): {T}")
        lines.append(f"CV: {len(list(self.cv_cols))} | MV: {len(list(self.mv_cols))}")
        lines.append(f"N (глобальный горизонт): {self.N}")
        lines.append(f"Позиция конца прогрева: {self.warmup_end_pos}")
        if getattr(self, "notes", None):
            lines.append("Заметки:")
            for note in self.notes:
                lines.append(f" • {note}")
        if getattr(self, "preprocessing", None):
            lines.append("Предобработка:")
            for key, value in self.preprocessing.items():
                lines.append(f" • {key}: {value}")
        lines.append("")

        lines.append("--- Open-loop (one-shot) ---")
        mean_r2_ol = np.round(self.openloop_R2_mean, digits) if not np.isnan(getattr(self, "openloop_R2_mean", np.nan)) else "nan"
        lines.append(f"Средний R² по CV, %: {mean_r2_ol}")

        ol_df = getattr(self, "openloop_metrics_overall", None)
        r2_col_ol = pick_r2_column(ol_df) if isinstance(ol_df, pd.DataFrame) else None
        if isinstance(ol_df, pd.DataFrame) and r2_col_ol:
            lines.append("R² по каждому CV, %:")
            lines.append(fmt_df(ol_df[[r2_col_ol]]))
            lines.append("")
            ol_sorted = ol_df.sort_values(r2_col_ol, ascending=False)
            best = ol_sorted.head(top_n)
            worst = ol_sorted.tail(top_n)
            lines.append(f"Топ-{top_n} CV по R² (open-loop):")
            lines.append(fmt_df(best[[r2_col_ol]]))
            lines.append("")
            lines.append(f"Худшие {top_n} CV по R² (open-loop):")
            lines.append(fmt_df(worst[[r2_col_ol]]))
            lines.append("")
        else:
            lines.append("R² по каждому CV: (нет данных)")
            lines.append("")

        lines.append("--- Rolling ---")
        if not getattr(self, "rolling_available", False):
            reason = getattr(self, "rolling_unavailable_reason", None) or "без указания причины"
            lines.append(f"Статус: недоступно — {reason}")
        else:
            lines.append("Статус: доступно")
            mean_r2_roll = np.round(self.rolling_R2_mean, digits) if not np.isnan(getattr(self, "rolling_R2_mean", np.nan)) else "nan"
            lines.append(f"Средний R² (усреднение по CV и горизонтам), %: {mean_r2_roll}")

            roll_overall = getattr(self, "rolling_metrics_overall", None)
            if isinstance(roll_overall, pd.DataFrame) and not roll_overall.empty:
                lines.append("Метрики по CV (усреднены по горизонтам) [sMAPE_%, R2_%]:")
                lines.append(fmt_df(roll_overall))
                lines.append("")
                r2_col_r = pick_r2_column(roll_overall)
                if r2_col_r:
                    ro_sorted = roll_overall.sort_values(r2_col_r, ascending=False)
                    best_r = ro_sorted.head(top_n)
                    worst_r = ro_sorted.tail(top_n)
                    lines.append(f"Топ-{top_n} CV по R² (rolling):")
                    lines.append(fmt_df(best_r[[r2_col_r]]))
                    lines.append("")
                    lines.append(f"Худшие {top_n} CV по R² (rolling):")
                    lines.append(fmt_df(worst_r[[r2_col_r]]))
                    lines.append("")
            else:
                lines.append("Метрики по CV (усреднены по горизонтам): (нет данных)")
                lines.append("")

            roll_h = getattr(self, "rolling_metrics_per_horizon", None)
            if isinstance(roll_h, pd.DataFrame) and not roll_h.empty:
                agg_h = (
                    roll_h.reset_index().groupby("horizon")[["sMAPE_%", "R2_%"]]
                    .mean().sort_index().round(digits)
                )
                lines.append("Усреднение по горизонтам (среднее по CV):")
                lines.append(agg_h.to_string())
                lines.append("")

        report = "\n".join(lines)
        if return_text:
            return report
        print(report)
        return None


# ───────────────────────── расчёт (без отрисовки) ─────────────────────────

@dataclass
class SSComputeParams:
    N: Union[int, str] = "max"     # 'min' | 'max' | int>0
    use_history: bool = False
    alpha_bias: float = 1.0
    dt: float = 1.0
    tau_is_steps: bool = True
    rolling_sample_pct: Optional[float] = None  # None ⇒ подобрать автоматически
    anchor_filter_len: int = 1         # 1 = без фильтра; >1 — MA по факту для якорения
    max_auto_N: Optional[int] = 200    # ограничение на авто-N при выборе 'max'


def _resolve_N(N_all: np.ndarray, N_req, max_auto: Optional[int]) -> tuple[int, bool]:
    capped = False
    if N_req == 'min':
        if N_all.size == 0:
            raise ValueError("N_all пуст; не из чего взять min().")
        return max(1, int(np.min(N_all))), capped
    if isinstance(N_req, (int, np.integer)) and N_req > 0:
        return int(N_req), capped
    if N_all.size == 0:
        raise ValueError("N_all пуст; не из чего взять max().")
    n_val = max(1, int(np.max(N_all)))
    if max_auto is not None and n_val > int(max_auto):
        n_val = int(max_auto)
        capped = True
    return n_val, capped


def _auto_rolling_sample_pct(T: int, n_cv: int, n_mv: int, N: int) -> tuple[float, int]:
    """Подбор rolling_sample_pct в зависимости от размера задачи."""

    if T <= 0:
        return 100.0, 0

    # Для коротких рядов считаем каждый такт — метрики тогда построены на всей
    # длине без дополнительных эвристик.
    if T <= max(2_000, 5 * max(1, N)):
        return 100.0, T

    eff_horizon = max(1.0, N / 50.0)
    eff_cv = max(1.0, n_cv / 4.0)
    eff_mv = 1.0 + 0.2 * max(0.0, (n_mv / 6.0) - 1.0)
    complexity = eff_horizon * eff_cv * eff_mv

    # «Бюджет» стартов rolling, который хотим выдерживать даже на крупных
    # задачах. При N≈200 это эквивалентно ~10 млн прогнозов CV.
    base_budget = 50_000.0
    approx_count = int(np.clip(base_budget / max(complexity, 1.0), 100, T))

    pct = float(np.clip(100.0 * approx_count / max(T, 1), 100.0 / T, 100.0))
    return pct, approx_count


def compute_forecast_ss(
    W, N_all, df: pd.DataFrame, mv_cols: Iterable[str], cv_cols: Iterable[str],
    p: SSComputeParams = SSComputeParams()
) -> ForecastResult:
    """
    Чистый расчёт open-loop и rolling. Без графики. Без авто-ужиманий N.
    Open-loop считается «на лету» во время прохода rolling (фиксируем bias_ol после прогрева).
    Rolling можно прореживать по проценту p.rolling_sample_pct.
    Якорение (bias) выполняется по фильтрованному факту z̄_t = MA(Z[t-n+1..t]), окно n=p.anchor_filter_len.
    """
    # проверки входа
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df пуст или не DataFrame.")

    mv_cols = list(mv_cols)
    cv_cols = list(cv_cols)
    for c in mv_cols + cv_cols:
        if c not in df.columns:
            raise ValueError(f"Колонка '{c}' отсутствует в df.")

    df = df.sort_index()
    U = df[mv_cols].to_numpy()
    Z = df[cv_cols].to_numpy()
    T = len(df)
    n_cv = len(cv_cols)
    n_mv = len(mv_cols)
    if T < 3:
        raise ValueError(f"Слишком мало точек: T={T}, нужно ≥ 3.")

    N_all = np.asarray(N_all, int)
    N, capped = _resolve_N(N_all, p.N, getattr(p, "max_auto_N", None))

    notes: list[str] = []
    preprocessing_info: Dict[str, Any] = {}
    user_messages: list[str] = []

    def _log(msg: str, *, store: bool = True) -> None:
        """Печатает сообщение сразу и при необходимости сохраняет его."""
        if store:
            user_messages.append(msg)
        print(msg, flush=True)

    _log(
        f"Исходные данные: T={T} (CV={n_cv}, MV={n_mv})"
    )
    horizon_msg = f"Выбранный горизонт N: {N}"
    if capped:
        max_auto = getattr(p, "max_auto_N", None)
        cap_note = (
            f"Горизонт N ограничен до {N} из-за max_auto_N={max_auto}."
        )
        notes.append(cap_note)
        horizon_msg += f" (ограничен max_auto_N={max_auto})"
    _log(horizon_msg)

    # ── создаём единый предиктор (будем из него «доставать» open-loop)
    pred = PredictSSRolling(W=W, N_all=N_all, dt=p.dt,
                            alpha_bias=p.alpha_bias, tau_is_steps=p.tau_is_steps, N=N)

    if len(U[1:1+N]) < N:
        raise ValueError(f"Недостаточно U для инициализации: {len(U[1:1+N])} < N={N} (T={T}).")

    # инициализация (z_t — фильтрованный факт на t=0)
    z0_f = _z_ma_inclusive(Z, t=0, win=p.anchor_filter_len)
    pred.initialize_at_t(u_t=U[0], z_t=z0_f, u_future=U[1:1+N],
                         u_past=None, warmup_len=None, assume_steady=not p.use_history)

    warmup_end_pos = N if (p.use_history and T > N) else 0
    if p.use_history and T > N:
        _log(
            f"Прогрев по факту включён: warmup_end_pos={warmup_end_pos}."
        )
    else:
        status = "включён" if p.use_history else "отключён"
        _log(
            f"Прогрев по факту {status}; warmup_end_pos={warmup_end_pos}."
        )

    # warmup по факту — тоже через фильтрованный факт
    if p.use_history and T > N:
        for t in range(0, N):
            z_next_f = _z_ma_inclusive(Z, t=t+1, win=p.anchor_filter_len)
            u_tail = U[t+N+1] if (t+N+1) < T else U[-1]
            _ = pred.advance(z_next=z_next_f, u_tail=u_tail)  # прогрев по фильтрованному факту

    # фиксируем open-loop bias после прогрева
    bias_ol = pred.bias.copy()
    Z_hat = np.full((T, n_cv), np.nan)

    # первая доступная точка open-loop: прогноз на t = warmup_end_pos → t+1
    if warmup_end_pos + 1 < T:
        yF = pred._rollout_from_now(pred._future_queue)   # (N, n_cv)
        Z_hat[warmup_end_pos + 1] = yF[0] + bias_ol

    # ── rolling возможность?
    rolling_available = True
    rolling_reason = None
    if T <= N:
        rolling_available = False
        rolling_reason = f"rolling невозможен: длина окна T={T} ≤ N={N}. Уменьшите N или расширьте окно данных."

    # прореживание
    pct_raw = getattr(p, "rolling_sample_pct", None)
    if isinstance(pct_raw, str):
        if pct_raw.strip().lower() == "auto":
            pct_raw = None
        else:
            raise ValueError("rolling_sample_pct должен быть числом, None или 'auto'.")

    approx_count = 0
    pct_source = "manual"
    if pct_raw is None:
        pct, approx_count = _auto_rolling_sample_pct(T=T, n_cv=n_cv, n_mv=n_mv, N=N)
        pct_source = "auto"
    else:
        pct = float(np.clip(pct_raw, 0.0, 100.0))
        approx_count = int(T if pct >= 99.5 else max(1, round(T * pct / 100.0))) if pct > 0 else 1

    pct = float(np.clip(pct, 0.0, 100.0))
    take_all = pct >= 99.5
    step = 1 if take_all else max(1, int(round(100.0 / max(pct, 1e-9))))

    approx_step = 1 if take_all else max(1, int(round(T / max(approx_count, 1))))

    sampling_details = {
        "mode": pct_source,
        "pct": pct,
        "approx_starts": int(approx_count),
        "approx_step": int(approx_step),
    }
    if pct_source == "auto":
        sampling_details.update({
            "T": int(T),
            "n_cv": int(n_cv),
            "n_mv": int(n_mv),
            "N": int(N),
        })
        auto_msg = (
            "rolling_sample_pct автоматически подобран: "
            f"≈{pct:.3f}% (≈{approx_count} стартов из {T}, шаг ≈ {approx_step}). "
            f"Параметры задачи: CV={n_cv}, MV={n_mv}, N={N}."
        )
        notes.append(auto_msg)
        _log(
            f"rolling_sample_pct (auto): ≈{pct:.3f}% (≈{approx_count} стартов, шаг≈{approx_step})."
        )
    elif pct < 99.5:
        manual_msg = (
            "rolling_sample_pct задан вручную: "
            f"≈{pct:.3f}% (≈{approx_count} стартов из {T}, шаг ≈ {approx_step})."
        )
        notes.append(manual_msg)
        _log(
            f"rolling_sample_pct (manual): ≈{pct:.3f}% (≈{approx_count} стартов, шаг≈{approx_step})."
        )
    else:
        _log(
            "rolling_sample_pct: 100% (rolling на каждом такте)."
        )

    preprocessing_info["rolling_sampling"] = sampling_details

    rolling_pred_by_horizon: Dict[int, pd.DataFrame] = {}
    rolling_metrics_per_horizon = pd.DataFrame(columns=["sMAPE_%", "R2_%"])
    rolling_metrics_overall = pd.DataFrame(columns=["sMAPE_%", "R2_%"])
    rolling_R2_mean = np.nan

    # массивы предсказаний по k (наполняем там, где считаем)
    preds_by_k = {k: np.full((T, n_cv), np.nan) for k in range(1, N+1)}

    # на t = warmup_end_pos уже можно записать «веер в ноль»
    zhat_win = pred._rollout_from_now(pred._future_queue)  # (N, n_cv)
    for k in range(1, N+1):
        preds_by_k[k][warmup_end_pos] = zhat_win[k-1] + pred.bias

    sampled_t: list[int] = []
    t_last = T - N - 1  # максимально возможный t, чтобы (t+N) был в пределах
    iter_idx = 0

    # основной проход (якоримся по фильтрованному факту на каждом шаге)
    for t in range(warmup_end_pos + 1, T):  # двигаемся до конца, чтобы построить OL ряд
        u_tail = U[t+N] if (t+N) < T else U[-1]
        z_next_f = _z_ma_inclusive(Z, t=t, win=p.anchor_filter_len)
        nowcast, y_next_model = pred.advance(z_next=z_next_f, u_tail=u_tail, return_model=True)

        # open-loop точка на t+1
        if (t + 1) < T:
            Z_hat[t + 1] = y_next_model + bias_ol

        # rolling «веера»/метрики — только на подвыборке тактов
        if rolling_available:
            do_sample = take_all or (iter_idx % step == 0)
            if do_sample and (t <= t_last):
                sampled_t.append(t)
                yF = pred._rollout_from_now(pred._future_queue)  # (N, n_cv)
                yF_roll = yF + pred.bias.reshape(1, -1)
                for k in range(1, N+1):
                    preds_by_k[k][t] = yF_roll[k-1]

        iter_idx += 1

    # упаковываем rolling предсказания по горизонтам
    sampling_details["actual_starts"] = int(len(sampled_t))
    if rolling_available:
        if sampled_t:
            _log(
                f"Rolling стартов рассчитано: {len(sampled_t)} (макс. t для rolling: {t_last})."
            )
        else:
            _log(
                "Rolling доступен, но подходящих стартов не найдено (проверьте warmup/N)."
            )
    else:
        reason_text = rolling_reason or "без указания причины"
        _log(f"Rolling отключён: {reason_text}")

    if not rolling_available:
        sampling_details["status"] = "unavailable"
        sampling_details["reason"] = reason_text
    else:
        sampling_details["status"] = "available"

    if rolling_available:
        for k in range(1, N+1):
            mask = np.isfinite(preds_by_k[k]).all(axis=1)
            if not np.any(mask):
                rolling_pred_by_horizon[k] = pd.DataFrame(columns=list(cv_cols))
            else:
                idxk = df.index[mask]
                rolling_pred_by_horizon[k] = pd.DataFrame(preds_by_k[k][mask], index=idxk, columns=list(cv_cols))

    # ── метрики open-loop (ВСЕГДА по сырому факту, без фильтра!)
    openloop_pred = pd.DataFrame(Z_hat, index=df.index, columns=list(cv_cols))
    eval_from = min(T, warmup_end_pos + 1)
    df_eval = df.iloc[eval_from:]
    zhat_eval = openloop_pred.iloc[eval_from:]

    rows = []
    for cv in cv_cols:
        sm, r2 = _compute_smape_r2(df_eval[cv].to_numpy(), zhat_eval[cv].to_numpy())
        rows.append({"cv": cv, "sMAPE_%": sm, "R2_%": r2})
    openloop_metrics_overall = pd.DataFrame(rows).set_index("cv") if rows else pd.DataFrame()
    openloop_R2_mean = float(openloop_metrics_overall["R2_%"].mean()) if not openloop_metrics_overall.empty else np.nan

    # ── метрики rolling: по sampled_t (тоже по сырому факту)
    if rolling_available and sampled_t:
        metrics_rows = []
        t_idx = np.array(sampled_t, dtype=int)
        for k in range(1, N+1):
            valid = []
            for t in t_idx:
                if (t + k) < T and np.all(np.isfinite(preds_by_k[k][t])):
                    valid.append(t)
            if not valid:
                continue
            valid = np.array(valid, dtype=int)
            yhat = preds_by_k[k][valid]
            y = Z[valid + k]  # сырой факт!
            for icv, cv in enumerate(cv_cols):
                sm, r2 = _compute_smape_r2(y[:, icv], yhat[:, icv])
                metrics_rows.append({"horizon": k, "cv": cv, "sMAPE_%": sm, "R2_%": r2})

        if metrics_rows:
            mph = pd.DataFrame(metrics_rows).set_index(["horizon", "cv"]).sort_index()
            overall_rows = []
            for cv in cv_cols:
                mcv = mph.xs(cv, level="cv")
                overall_rows.append({"cv": cv,
                                     "sMAPE_%": mcv["sMAPE_%"].mean(),
                                     "R2_%":    mcv["R2_%"].mean()})
            rolling_metrics_per_horizon = mph
            rolling_metrics_overall = pd.DataFrame(overall_rows).set_index("cv")
            rolling_R2_mean = float(rolling_metrics_overall["R2_%"].mean())

    if notes:
        for note in notes:
            if note not in user_messages:
                _log(note)

    return ForecastResult(
        df=df,
        mv_cols=mv_cols,
        cv_cols=cv_cols,
        openloop_pred=openloop_pred,
        openloop_metrics_overall=openloop_metrics_overall,
        openloop_R2_mean=openloop_R2_mean,
        rolling_pred_by_horizon=rolling_pred_by_horizon,
        rolling_metrics_per_horizon=rolling_metrics_per_horizon,
        rolling_metrics_overall=rolling_metrics_overall,
        rolling_R2_mean=rolling_R2_mean,
        N=N,
        warmup_end_pos=warmup_end_pos,
        rolling_available=rolling_available,
        rolling_unavailable_reason=rolling_reason,
        notes=notes,
        preprocessing=preprocessing_info,
    )
