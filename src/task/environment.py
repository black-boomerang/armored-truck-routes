import numpy as np


class Environment:
    """Ограничения и данные задачи"""

    def __init__(
            self,
            non_cashable_loss_percent_per_year=0.02,
            cashable_loss_percent=0.01 / 100,
            cashable_min_loss=100,
            non_serviced_days=14,
            terminal_limit=1_000_000,
            armored_price=20_000,
            working_day_time=12 * 60,
            encashment_time=10
        ):

        self.non_cashable_loss_percent = non_cashable_loss_percent_per_year / 365
        self.cashable_loss_percent = cashable_loss_percent
        self.cashable_min_loss = cashable_min_loss
        self.non_serviced_days = non_serviced_days
        self.terminal_limit = terminal_limit
        self.armored_price = armored_price
        self.working_day_time = working_day_time
        self.encashment_time = encashment_time

    def get_non_cashable_loss(self, cash: float):
        return self.non_cashable_loss_percent * cash

    def get_cashable_loss(self, cash: float):
        return np.maximum(self.cashable_loss_percent * cash, self.cashable_min_loss)
