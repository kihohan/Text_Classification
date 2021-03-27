import re

import tqdm
from multiprocessing import Pool, Manager
from multiprocessing.dummy import Pool as ThreadPool, Manager as ThreadManager

class mp():
    '''
    multi-processing interface
    '''
    def __init__(self, task_func, param_list, n_process=4, using_tqdm=False, tqdm_desc='mp2'):
        self.task_func = task_func
        self.param_list = param_list
        self.n_process = n_process
        self.using_tqdm = using_tqdm
        self.tqdm_desc = tqdm_desc
        
    def run(self):
        self.result = []
        with Pool(self.n_process) as pool, Manager() as manager:
            # communication channel between process / with parent
            # predefined codes
            # 1. 'break' : True - stop all children and return 
            comm = manager.dict()
            self.param_list = [(comm, p) for p in self.param_list]
            
            if self.using_tqdm: # first come first out. ie. unordered result
                for r in tqdm.tqdm(pool.imap_unordered(self.task_func, self.param_list), total=len(self.param_list), desc=self.tqdm_desc):
                    self.result.append(r)
            else: # ordered result
                for r in pool.map(self.task_func, self.param_list):
                    self.result.append(r)
                    
        return self.result
    
def clean_spm(lst):
    def clean_text(text):
        return re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
    def clean_num(text):
        return re.sub('\d+', '', text)
    def _del(text):
        return text.replace('▁','')
    a = [clean_text(x) for x in lst] 
    b = [clean_num(x) for x in a] 
    c = [_del(x) for x in b]
    d = [x for x in c if len(x) != 1]
    e = ['즉석죽' if x=='죽' else x for x in d]
    f = ['껌껌' if x=='껌' else x for x in e]
    g = ['Tea' if x=='티' else x for x in f]
    h = [x for x in g if len(x) != 1]
    return h
