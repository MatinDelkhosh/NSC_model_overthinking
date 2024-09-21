class LossHyperparameters:
    def __init__(self, βv, βe, βp, βr):
        self.βv = float(βv)
        self.βe = float(βe)
        self.βp = float(βp)
        self.βr = float(βr)

# Example usage:
loss_hp = LossHyperparameters(βv=0.1, βe=0.2, βp=0.3, βr=0.4)