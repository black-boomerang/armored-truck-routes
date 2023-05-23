class BusinessLogic:
    """ Бизнес-ограничения и данные задачи"""

    def __init__(self):
        self.non_cashable_loss_percent = 0.02 / 365
        self.cashable_loss_percent = 0.01 / 100
        self.cashable_min_loss = 100
        self.non_serviced_days = 14
        self.terminal_limit = 1_000_000
        self.armored_price = 20_000
        self.working_day_time = 12 * 60
        self.encashment_time = 10

    def get_non_cashable_loss(self, cash: float):
        return self.non_cashable_loss_percent * cash

    def get_cashable_loss(self, cash: float):
        return max(self.cashable_loss_percent * cash, self.cashable_min_loss)
