def find_path(tree):
    path = []
    for nodes in reversed(tree):
        if len(path) == 0:
            path.append(nodes[0])
        else:
            parent_id = path[-1].parent_id
            for node in nodes:
                if node.id == parent_id:
                    path.append(node)
    return path

def find_best_path(tree):
    best = []
    for nodes in reversed(tree):
        if len(best) == 0:
            best.append(nodes[0])
        else:
            nodes_eos = []
            parent_id = best[-1].parent_id
            for node in nodes:
                if node.eos:
                    nodes_eos.append(node)
                if node.id == parent_id:
                    best.append(node)
            if len(nodes_eos) > 0:
                candidates = sorted([best[-1], *nodes_eos],
                                    key=lambda node: node.logps,
                                    reverse=True)
                candidate = candidates[0]
                if candidate.eos:
                    best = [candidate]
    return best

class Node:
    id_ = 0
    
    def __init__(self, token, states, logp=0., parent=None, eos=False):
        self.__id = self.__class__.id_
        self.__token = token
        self.__states = states
        self.__logp = logp
        self.__parent_id = None if parent is None else parent.id
        self.__eos = eos
        self.__level = 0 if parent is None else parent.level + 1
        self.__logps = logp if parent is None else parent.logps + logp
        self.__class__.id_ += 1
        
    def __str__(self):
        return f'Node[id={self.__id}, ' + \
                    f'index={EN.vocab.itos[self.__token.cpu().item()]}, ' + \
                    f'logp={self.__logp}, ' + \
                    f'logps={self.__logps}, ' + \
                    f'parent_id={self.__parent_id}, ' + \
                    f'level={self.__level}]'
    
    @property
    def token(self):
        return self.__token
    
    @token.setter
    def token(self, token):
        self.__token = token
    
    @property
    def parent_id(self):
        return self.__parent_id
    
    @parent_id.setter
    def parent_id(self, parent_id):
        self.__parent_id = parent_id
        
    @property
    def id(self):
        return self.__id
    
    @id.setter
    def id(self, id_):
        self.__id = id_
    
    @property
    def token(self):
        return self.__token
    
    @token.setter
    def token(self, token):
        self.__token = token
    
    @property
    def states(self):
        return self.__states
    
    @states.setter
    def states(self, states):
        self.__states = states
      
    @property
    def eos(self):
        return self.__eos
    
    @eos.setter
    def eos(self, eos):
        self.__eos = eos
    
    @property
    def logps(self):
        return self.__logps
    
    @logps.setter
    def logps(self, logps):
        self.__logps = logps
        
    @property
    def level(self):
        return self.__level
    
    @level.setter
    def level(self, level):
        self.__level = level
