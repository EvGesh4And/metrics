import numpy as np
from scipy.signal import cont2discrete, tf2ss


# ---------- одна SISO-связь MV_j -> CV_i ----------
class LinkSS:
    def __init__(self, A, B, C, D, delay_steps=0):
        self.A = np.asarray(A)
        self.B = np.asarray(B).reshape(-1)
        self.C = np.asarray(C).reshape(-1)
        self.D = float(np.asarray(D).squeeze())
        self.x = np.zeros(self.A.shape[0])
        self.delay = int(delay_steps)
        if self.delay > 0:
            self.buf = np.zeros(self.delay)
            self.ptr = 0
        else:
            self.buf = None
            self.ptr = 0

    def _push_delay(self, u):
        if self.buf is None:
            return u
        out = self.buf[self.ptr]
        self.buf[self.ptr] = u
        self.ptr = (self.ptr + 1) % self.delay
        return out

    def step(self, u):
        ue = self._push_delay(float(u))
        self.x = self.A @ self.x + self.B * ue
        return float(self.C @ self.x + self.D * ue)

    def rollout(self, u_seq):
        u_seq = np.asarray(u_seq, float)
        K = len(u_seq)
        y = np.zeros(K)
        x = self.x.copy()
        if self.buf is None:
            for k in range(K):
                u = float(u_seq[k])
                x = self.A @ x + self.B * u
                y[k] = float(self.C @ x + self.D * u)
        else:
            buf = self.buf.copy()
            ptr = self.ptr
            for k in range(K):
                u = float(u_seq[k])
                ue = buf[ptr]
                buf[ptr] = u
                ptr = (ptr + 1) % self.delay
                x = self.A @ x + self.B * ue
                y[k] = float(self.C @ x + self.D * ue)
        return y

    # ---- выход в текущий такт без модификации состояния/буфера ----
    def output_now(self, u):
        ue = float(u) if self.buf is None else float(self.buf[self.ptr])
        return float(self.C @ self.x + self.D * ue)

    # ---- стационарная инициализация по u ----
    def set_steady_state(self, u, fallback_steps=30):
        u = float(u)
        if self.buf is not None:
            self.buf[:] = u
            self.ptr = 0
            ue = u
        else:
            ue = u
        I = np.eye(self.A.shape[0])
        M = I - self.A
        b = self.B * ue
        try:
            self.x = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            # интегратор/квази-интегратор: мягкий прогрев
            self.x[:] = 0.0
            for _ in range(int(max(1, fallback_steps))):
                _ = self.step(u)
            if self.buf is not None:
                self.ptr = (self.ptr - 1) % self.delay


# ---------- сборка звена из W ----------
def _tf_cont_from_cell(cell):
    K = float(cell['K'])
    num = np.asarray(cell['num'], float) * K
    den = np.asarray(cell['denum'], float)
    if cell.get('integrator', False):
        den = np.concatenate([den, [0.0]])  # множитель 1/s
    return num, den

def _discretize_cell(cell, dt, tau_is_steps=True):
    num_c, den_c = _tf_cont_from_cell(cell)
    num_z, den_z, _ = cont2discrete((num_c, den_c), dt=dt, method="zoh")
    num_z = np.asarray(num_z).squeeze()
    den_z = np.asarray(den_z).squeeze()
    A, B, C, D = tf2ss(num_z, den_z)
    tau = float(cell.get('tau', 0.0))
    delay = int(round(tau if tau_is_steps else tau / dt))
    return A, B, C, D, max(0, delay)


# ---------- предиктор с глобальным горизонтом ----------
class PredictSSRolling:
    """
    Оценка/прогноз на двухстороннем окне с глобальным горизонтом:
      • N_global = min/max/конкретное число (единый для всех CV).
      • initialize_at_t(...) принимает либо u_past (историю), либо только u_t:
            если u_past=None, стационарная инициализация по u_t (fallback: прогрев).
      • forecast_current() -> (N_global, n_cv).
      • advance(z_{t+1}, u_{t+N+1}) — сдвиг окна; опционально возвращает (nowcast, y_next_model).
    """
    def __init__(self, W, N_all, dt=1.0, alpha_bias=1.0, tau_is_steps=True, N: int | str = 'max'):
        # --- размеры ---
        self.W = W
        self.n_cv = len(W)
        self.n_mv = len(W[0]) if self.n_cv else 0
        self.dt = float(dt)
        self.alpha = float(alpha_bias)
        self.tau_is_steps = bool(tau_is_steps)

        # --- глобальный горизонт ---
        N_all = np.asarray(N_all, int)
        if isinstance(N, (int, np.integer)) and N > 0:
            self.N_global = int(N)
        elif N == 'min':
            if N_all.size == 0:
                raise ValueError("N_all пустой, не из чего брать min()")
            self.N_global = max(1, int(N_all.min()))
        else:  # 'max' по умолчанию
            if N_all.size == 0:
                raise ValueError("N_all пустой, не из чего брать max()")
            self.N_global = max(1, int(N_all.max()))

        # --- звенья ---
        self.links = [[None] * self.n_mv for _ in range(self.n_cv)]
        for i in range(self.n_cv):
            for j in range(self.n_mv):
                cell = W[i][j]
                if cell is None:
                    continue
                A, B, C, D, delay = _discretize_cell(cell, dt=self.dt, tau_is_steps=self.tau_is_steps)
                self.links[i][j] = LinkSS(A, B, C, D, delay_steps=delay)

        # --- якорение по факту ---
        self.bias = np.zeros(self.n_cv)

        # --- очередь будущих u длины N_global ---
        self._future_queue = None  # shape (N_global, n_mv)

    # -------- служебные --------
    @property
    def N(self) -> int:
        return self.N_global

    def _model_step_now(self, u_vec):
        u = np.asarray(u_vec, float).reshape(-1)
        y = np.zeros(self.n_cv)
        for i in range(self.n_cv):
            acc = 0.0
            for j in range(self.n_mv):
                link = self.links[i][j]
                if link is not None:
                    acc += link.step(u[j])
            y[i] = acc
        return y

    def _rollout_from_now(self, u_future):
        uf = np.asarray(u_future, float)
        K = uf.shape[0]
        yF = np.zeros((K, self.n_cv))
        for i in range(self.n_cv):
            acc = np.zeros(K)
            for j in range(self.n_mv):
                link = self.links[i][j]
                if link is not None:
                    acc += link.rollout(uf[:, j])
            yF[:, i] = acc
        return yF

    # -------- API --------
    def initialize_at_t(self, u_t, z_t, u_future, u_past=None, warmup_len=None, assume_steady=True):
        """
        Инициализация в моменте t.

        u_t      : (n_mv,) — текущий вход u_t.
        z_t      : (n_cv,) — факт в момент t (для якорения).
        u_future : (>=N_global, n_mv) — будущие u_{t+1..}; обрежется до N_global.
        u_past   : (L, n_mv) или None — история u_{t-L+1..t}. Если None:
                    - если assume_steady=True: стационарная инициализация по u_t;
                    - иначе: прогрев L шагов на u_t (L = warmup_len или N_global).
        """
        u_t = np.asarray(u_t, float).reshape(-1)
        z_t = np.asarray(z_t, float).reshape(-1)
        uf = np.asarray(u_future, float)
        assert u_t.shape[0] == self.n_mv and z_t.shape[0] == self.n_cv
        assert uf.ndim == 2 and uf.shape[1] == self.n_mv

        if u_past is not None:
            up = np.asarray(u_past, float)
            assert up.ndim == 2 and up.shape[1] == self.n_mv
            for r in range(up.shape[0]):
                _ = self._model_step_now(up[r])
        else:
            if assume_steady:
                for i in range(self.n_cv):
                    for j in range(self.n_mv):
                        link = self.links[i][j]
                        if link is not None:
                            link.set_steady_state(u_t[j], fallback_steps=self.N_global)
            else:
                L = self.N_global if warmup_len is None else int(max(1, warmup_len))
                for _ in range(L):
                    _ = self._model_step_now(u_t)

        # модельный выход в t БЕЗ шага
        y_t_model = np.zeros(self.n_cv)
        for i in range(self.n_cv):
            s = 0.0
            for j in range(self.n_mv):
                link = self.links[i][j]
                if link is not None:
                    s += link.output_now(u_t[j])
            y_t_model[i] = s

        # якорение: подгоняем уровень под факт
        self.bias = (1.0 - self.alpha) * self.bias + self.alpha * (z_t - y_t_model)

        # очередь будущих входов ровно длины N_global
        if uf.shape[0] < self.N_global:
            raise ValueError(f"u_future слишком короткий: {uf.shape[0]} < N_global={self.N_global}")
        self._future_queue = uf[:self.N_global].copy()

    def forecast_current(self):
        """
        Прогноз из точки t на N_global шагов: вернёт (N_global, n_cv) с учётом текущего bias.
        """
        if self._future_queue is None:
            raise RuntimeError("Сначала вызовите initialize_at_t(...)")
        yF = self._rollout_from_now(self._future_queue)
        return yF + self.bias.reshape(1, -1)

    def advance(self, z_next, u_tail, return_model: bool = False):
        """
        Сдвиг окна t -> t+1 для оценки качества:
          • реальный шаг на u_{t+1} = first(self._future_queue)
          • якоримся по факту z_{t+1}
          • сдвигаем очередь и добавляем u_tail = u_{t+N+1}

        Возвращает:
          - если return_model=False (по умолчанию): nowcast (shape (n_cv,))
          - если return_model=True: (nowcast, y_next_model), где
                y_next_model — модельный выход БЕЗ прибавки bias (удобно для open-loop).
        """
        if self._future_queue is None:
            raise RuntimeError("Сначала вызовите initialize_at_t(...)")

        z_next = np.asarray(z_next, float).reshape(-1)
        u_tail = np.asarray(u_tail, float).reshape(-1)
        assert z_next.shape[0] == self.n_cv and u_tail.shape[0] == self.n_mv

        # 1) реальный шаг модели на u_{t+1}
        u_next = self._future_queue[0]
        y_next_model = self._model_step_now(u_next)  # НЕТ bias

        # 2) якорение по факту z_{t+1}
        err = z_next - y_next_model
        self.bias = (1.0 - self.alpha) * self.bias + self.alpha * err
        nowcast = y_next_model + self.bias  # уже С учётом bias

        # 3) обновить очередь будущих u (сдвиг + хвост)
        self._future_queue = np.vstack([self._future_queue[1:], u_tail])

        if return_model:
            return nowcast, y_next_model
        return nowcast
