class FakeTrial:
    def __init__(self, optuna_params):
        super(FakeTrial, self).__init__()
        self.optuna_params = optuna_params

    def suggest_categorical(self, name, *args,**vargs):
        return self.get_param(name)

    def suggest_float(self, name, *args,**vargs):
        return self.get_param(name)

    def suggest_int(self, name, *args,**vargs):
        return self.get_param(name)

    def get_param(self, name):
        return self.optuna_params[name]
