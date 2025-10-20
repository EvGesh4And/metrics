from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

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

    # ----------------- ФАКТ vs OPEN-LOOP -----------------
    def plot_openloop(self, start_pos: int | None = None, end_pos: int | None = None,
                      title_prefix: str = "Open-loop"):
        """
        Факт vs open-loop на [start_pos : end_pos) (end_pos эксклюзивный).
        Серая зона прогрева и «дырка» в прогнозе строятся по пересечению окна с [0:warmup_end_pos).
        """
        n = len(self.df)
        s = 0 if start_pos is None else int(start_pos)
        e = n if end_pos   is None else int(end_pos)
        if not (0 <= s < n) or not (0 < e <= n) or not (s < e):
            raise ValueError(f"Неверный диапазон: start_pos={s}, end_pos={e}, len={n}. "
                             "Должно быть: 0 <= s < e <= len(df).")

        dfw = self.df.iloc[s:e]
        zhw = self.openloop_pred.loc[dfw.index]
        L = len(dfw)
        x = np.arange(L)

        warmup_global = int(self.warmup_end_pos)
        warm_len_local = max(0, min(warmup_global, e) - s)

        for cv in self.cv_cols:
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

    # ---------------------- «ВЕЕР» ROLLING ----------------------
    def plot_rolling(self, start_pos: int | None = None, end_pos: int | None = None,
                     fan_stride: int = 100, fan_max: int = 20,
                     title_prefix: str = "Rolling"):
        """
        «Веер» rolling на [start_pos : end_pos) (end_pos эксклюзивный).
        Серая зона прогрева и запрет стартов внутри неё строятся по пересечению окна с [0:warmup_end_pos).
        """
        if not self.rolling_available:
            reason = self.rolling_unavailable_reason or "rolling не рассчитан."
            raise ValueError(f"Нельзя построить веер: {reason}")

        base_df = self.rolling_pred_by_horizon.get(1)
        if base_df is None or base_df.empty:
            raise ValueError("Нельзя построить веер: нет предсказаний для горизонта k=1.")

        n = len(self.df)
        s = 0 if start_pos is None else int(start_pos)
        e = n if end_pos   is None else int(end_pos)
        if not (0 <= s < n) or not (0 < e <= n) or not (s < e):
            raise ValueError(f"Неверный диапазон: start_pos={s}, end_pos={e}, len={n}. "
                             "Должно быть: 0 <= s < e <= len(df).")

        dfw = self.df.iloc[s:e]
        L = len(dfw)
        x = np.arange(L)

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

        for cv in self.cv_cols:
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

    def summary(self, digits: int = 1, top_n: int = 3, return_text: bool = False) -> str | None:
        """
        Печатает (и/или возвращает) краткий отчёт по расчёту:
        • размеры данных, N и позиция прогрева;
        • open-loop (one-shot): средний R² и таблица R² по CV;
        • rolling: статус/причина, средний R², агрегат по CV и (опционально) срез по горизонтам;
        • топ-CV по R² (лучшие/худшие) для open-loop и rolling.
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
    rolling_sample_pct: float = 100.0  # 0..100 — доля тактов, где считаем «веер»/метрики rolling
    anchor_filter_len: int = 1         # 1 = без фильтра; >1 — MA по факту для якорения


def _resolve_N(N_all: np.ndarray, N_req) -> int:
    if N_req == 'min':
        if N_all.size == 0:
            raise ValueError("N_all пуст; не из чего взять min().")
        return max(1, int(np.min(N_all)))
    if isinstance(N_req, (int, np.integer)) and N_req > 0:
        return int(N_req)
    if N_all.size == 0:
        raise ValueError("N_all пуст; не из чего взять max().")
    return max(1, int(np.max(N_all)))


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
    for c in list(mv_cols) + list(cv_cols):
        if c not in df.columns:
            raise ValueError(f"Колонка '{c}' отсутствует в df.")

    df = df.sort_index()
    U = df[mv_cols].to_numpy()
    Z = df[cv_cols].to_numpy()
    T = len(df); n_cv = len(list(cv_cols))
    if T < 3:
        raise ValueError(f"Слишком мало точек: T={T}, нужно ≥ 3.")

    N_all = np.asarray(N_all, int)
    N = _resolve_N(N_all, p.N)

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
    pct = float(np.clip(p.rolling_sample_pct, 0.0, 100.0))
    take_all = pct >= 99.5
    step = 1 if take_all else max(1, int(round(100.0 / max(pct, 1e-9))))

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

    return ForecastResult(
        df=df,
        mv_cols=list(mv_cols),
        cv_cols=list(cv_cols),
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
        rolling_unavailable_reason=rolling_reason
    )
