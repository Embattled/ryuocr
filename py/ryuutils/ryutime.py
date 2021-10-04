import time

class TimeMemo:
    def __init__(self):
        self.t=time.time()

    def reset(self):
        self.t=time.time()
    
    def getNowTimeStr(self,gap='-'):
        if gap != '-':
            fm="%Y"+gap+"%m"+gap+"%d"+gap+"%H"+gap+"%M"+gap+"%S"
            return time.strftime(fm)
        return time.strftime("%Y-%m-%d-%H-%M-%S")

    def getTimeCostStr(self)->str:
        e=time.time()
        second=(int)(e-self.t)

        hour=second//3600
        second%=3600
        minute=second//60
        second%=60

        s=(str(hour)+" h " if hour>0 else '') + (str(minute)+" m " if minute>0 else '') + str(second)+" s"
        return s

    def getTimeCost(self):
        e=time.time()
        second=(int)(e-self.t)

        hour=second//3600
        second%=3600
        minute=second//60
        second%=60

        return second,minute,hour

if __name__ == "__main__":

    t=TimeMemo()
    print(t.nowTimeStr())
    time.sleep(3)

    print(t.getRangeStr())

