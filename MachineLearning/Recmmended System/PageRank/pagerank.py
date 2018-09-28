from pygraph.classes.digraph import digraph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class pageRank(object):

    def __init__(self, dg):
        self.alpha = 0.85
        self.maxCycles = 200
        self.min_delta = 0.0001
        self.graph = dg

    def page_rank(self):
        #没有出链的点先加上和所有点的边
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    digraph.add_edge(self.graph, (node, node2))
        nodes = self.graph.nodes()
        graphs_size = len(nodes)

        if graphs_size == 0:
            return 'nodes set is empty!'

        page_rank = dict.fromkeys(nodes, 1.0/graphs_size)
        runAway = (1.0 - self.alpha) / graphs_size
        flag = False
        for i in range(self.maxCycles):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):
                    rank += self.alpha * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
                rank += runAway
                change += abs(page_rank[node] - rank)
                page_rank[node] = rank

            print("NO.%s iteration" % (i + 1))
            print(page_rank)

            if change < self.min_delta:
                flag = True
                break
        return page_rank

if __name__ == '__main__':
    dg = digraph()

    dg.add_nodes(["A", "B", "C", "D", "E"])

    dg.add_edge(("A", "B"))
    dg.add_edge(("A", "C"))
    dg.add_edge(("A", "D"))
    dg.add_edge(("B", "D"))
    dg.add_edge(("C", "E"))
    dg.add_edge(("D", "E"))
    dg.add_edge(("B", "E"))
    dg.add_edge(("E", "A"))

    pr = pageRank(dg)
    page_ranks = pr.page_rank()

    print("The final page rank is\n", page_ranks)
