import yaml

def loadyaml(path:str)->dict:

    try:
        ymlfile =open(path,'r',encoding='utf-8')
        config = yaml.safe_load(ymlfile)
        return config
    except:
        return None
def saveyaml(conf:dict,path:str):
    with open(path,'a') as f:
        yaml.dump(conf,f,default_flow_style=False)

if __name__ == "__main__":

    pass