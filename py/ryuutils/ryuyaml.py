import yaml
import sys

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

def printyaml(config):
    print(yaml.dump(config,sys.stdout,default_flow_style=False))

def check_config(conf:dict)->dict:

    pass
if __name__ == "__main__":

    pass