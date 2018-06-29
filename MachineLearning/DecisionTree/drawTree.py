import matplotlib.pyplot as plt
class TreeNode(object):
    def __init__(self, x, y, parentX = None, parentY = None):
        self.x = x
        self.y = y
        self.parentX = parentX
        self.parentY = parentY
    pass

def getNumLeafs(myTree):
    if myTree == None:
        return 0
    elif myTree.right == None and myTree.left == None:
        return 1
    else:
        return getNumLeafs(myTree.right) + getNumLeafs(myTree.left)

def getDepth(myTree):
    if myTree == None:
        return 0
    right = getDepth(myTree.right)
    left = getDepth(myTree.left)
    return max(right+1, left+1)

def drawNode(x, y ,parent,color, marker, myTree, position):
    if myTree.results == None or len(list(myTree.results.keys())) > 1:
        plt.scatter(x, y, c=color, marker=marker, s=200)

    if myTree.right == None and myTree.left == None:
        results = list(myTree.results.keys())
        plt.annotate(s = 'label == ' + str(results[0]), xy=(x - 15, y))
        if results[0] == 0.0:
           plt.annotate(s='label == 0.0', xy=(x , y))
           plt.scatter(x, y, c='orange', marker='H', s=100)
        if results[0] == 1.0:
           plt.scatter(x, y, c='pink', marker='8', s=100)
        if results[0] == 2.0:
           plt.scatter(x, y, c='r', marker='+', s=100)

    if myTree.value != None and myTree.fea != None:
        po = 5
        if position == 'right':
           plt.annotate(s = 'dimension' + str(myTree.fea) + '>' + str(round(myTree.value, 2)), xy = (x-25 - po, y))
        else:
           plt.annotate(s='dimension' + str(myTree.fea) + '>' + str(round(myTree.value, 2)), xy=(x - 25 + po, y))
    if parent != None:
       plt.plot([x, parent.x], [y, parent.y], color = 'gray', alpha = 0.5)
def draw(myTree, parent = None, x = 100, y = 100, color = 'r', marker = '^', position = None):
    NumberLeaf = getNumLeafs(myTree)
    Depth = getDepth(myTree)
    delta = (NumberLeaf+Depth)
    drawNode(x, y, parent, color, marker, myTree,position)
    if myTree.right != None:
        draw(myTree.right, parent=TreeNode(x, y) ,x=x+5*delta, y=y-5-delta,color='b', marker='x', position='right')
    if myTree.left != None:
        draw(myTree.left,parent=TreeNode(x, y) ,x=x-5*delta, y=y-2-delta, color='g', marker='o', position='left')
    pass
