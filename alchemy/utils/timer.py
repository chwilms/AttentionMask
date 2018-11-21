import datetime

class Timer(object):
    
    def __init__(self):
        self.delta = 0
        self.start = 0
        self.end = 0
        pass

    def tic(self):
        self.start = datetime.datetime.now()
        return self.start

    def tac(self):
        self.end = datetime.datetime.now()
        self.delta = self.end - self.start
        return self.delta
