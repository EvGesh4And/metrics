import math

time_mapper = {
        'seconds': 1,
        'minutes': 60,
        'hours': 3600,
    }
    

class Plotmatrix:

    def __init__(self, MV_set, CV_set, DV_set, plot_raw) -> None:

        # lim - максимум графиков в ряд на странице
            
        # рассчет размеров матрицы графиков - ширины и длины

        # иначе рассчитываем из конфига
        width_max = max(len(CV_set), len(MV_set), len(DV_set))

        self.width = width_max if width_max < plot_raw else plot_raw
        self.length = math.ceil(len(CV_set)/self.width)

        count_mv_dv = len(MV_set)
        
        if DV_set:
            count_mv_dv += len(DV_set)

        self.length += math.ceil(count_mv_dv/self.width)

        # (можно решить корректнее) обход ошибки, если матрица графиков = 1:1
        if self.length == 1 and self.width == 1:
            self.length = 2

        self.plot_position = {'length': 0, 'width': 0} # позиция первого графика

    def lower_position(self):  # ф-я перевода положения графика на новую строку матрицы
        self.plot_position['length'] += 1
        self.plot_position['width'] = 0

    # обновляет позицию, сдвигает вправо или переходит на новую строку при превышении ширины 
    def update_position(self):

        if (self.plot_position['width'] + 1 >= self.width):
            self.lower_position()
        else:
            self.plot_position['width'] += 1
    
    def get_position(self): # возвращает координаты для отрисовки графиков через matplotlib

        if self.width > 1 and self.length > 1:
            position = (self.plot_position['length'], self.plot_position['width'])
        elif self.width == 1:
            position = self.plot_position['length']
        elif self.length == 1:
            position = self.plot_position['width']
        
        return position