import os.path as osp
import random
class Datasets():
    """
    node_label.txt
    node_atrri.txt
    edge_index.txt
    interactions.txt
    """    
    def node_label():
        pass
    def node_node_atrri():
        pass
    def edge_index():
        pass
    def interactions():
        pass
    
    
class Cora(Datasets):
    def __init__(self) -> None:
        super().__init__()
        self.path = "/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/origin/"
        self.pathw = "/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/raw/"
        self.areas = {
        "Case_Based":0,
		"Genetic_Algorithms":1,
		"Neural_Networks":2,
		"Probabilistic_Methods":3,
		"Reinforcement_Learning":4,
		"Rule_Learning":5,
		"Theory":6
        }
        
    def node_label(self):
        # read data
        with open(self.path +'cora.content', 'r') as f:
            data =  f.read().split('\n')
            if len(data[-1]) == 0:       # 如果最后一行为空，则去掉该行
                data.pop()
            f.close()
         
        # processed 
        line = []  
        for ele in data:
            line.append(ele.split('	'))
            
        # map
        # old_id[old] = new
        old_id = {}
        cnt = 0 
        for ele in line:
            old_id[int(ele[0])] = cnt
            cnt += 1
            
        with open(self.pathw +'old_id.txt', 'w') as fw:
            for key,value in old_id.items():
                fw.write(str(key))
                fw.write("	")
                fw.write(str(value))
                fw.write("\n")
            fw.close()
            
        with open(self.pathw +'label.txt', 'w') as fw:
            for ele in line:
                fw.write(str(self.areas[ele[-1]]))
                fw.write("\n")
            fw.close()
            
            
    def node_atrri(self):
        with open(self.path +'cora.content', 'r') as f:
            data =  f.read().split('\n')
            if len(data[-1]) == 0:       # 如果最后一行为空，则去掉该行
                data.pop()
            f.close()   
            
        line = []
        for ele in data:
            ele = ele.split("	")
            del(ele[0])
            del(ele[-1])
            line.append(ele)
            
        with open(self.pathw +'node_atrri.txt', 'w') as fw:
            for ele in line:
                lens = len(ele)
                for index, e in enumerate(ele):
                     fw.write(e)
                     if index  != lens -1:
                        fw.write('	')
                fw.write('\n')
            fw.close()
            
    def edge_index(self):
        old_id = {}
        with open(self.pathw +'old_id.txt', 'r') as f:
            data =  f.read().split('\n')
            if len(data[-1]) == 0:       # 如果最后一行为空，则去掉该行
                data.pop()
            f.close()   
        
        for line in data:
            line = line.split(	)
            old_id[int(line[0])] = int(line[-1])

        with open(self.path +'cora.cites', 'r') as f:
            data =  f.read().split('\n')
            if len(data[-1]) == 0:       # 如果最后一行为空，则去掉该行
                data.pop()
            f.close() 
        
            
        with open(self.pathw +'edge_index.txt', 'w') as fw:
            for line in data:
                line = line.split(	)
                fw.write(str(old_id[int(line[0])]))
                fw.write('	')
                fw.write(str(old_id[int(line[-1])]))
                fw.write('\n')
            fw.close()
    
    def interactions(self):
        with open(self.pathw +'edge_index.txt', 'r') as f:
            data =  f.read().split('\n')
            if len(data[-1]) == 0:       # 如果最后一行为空，则去掉该行
                data.pop()
            f.close()   
         
        with open(self.pathw +'interactions.txt', 'w') as fw:   
            for line in data:
                line = line.split(	)
                number = random.randint(0,200)
                fw.write(str(number))
                fw.write('\n')
            fw.close()
            
if __name__ == "__main__":
    cora = Cora()
    cora.node_label()
    cora.node_atrri()
    cora.edge_index()
    cora.interactions()
    
                
            
        
        
            
    