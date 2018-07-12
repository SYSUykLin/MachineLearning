import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

def loadDataSet(filename):
    '''
    load dataSet
    :param filename: the filename which you need to open
    :return: dataset in file
    '''
    dataMat = pd.read_csv(filename)
    for i in range(np.shape(dataMat)[0]):
        if dataMat.iloc[i, 2] == 0:
            dataMat.iloc[i, 2] = -1
    return dataMat
    pass

def split_data(data_array, col, value):
    '''split the data according to the feature'''
    array_1 = data_array.loc[data_array.iloc[:, col] >= value, :]
    array_2 = data_array.loc[data_array.iloc[:, col] < value, :]
    return array_1, array_2
    pass

def getErr(data_array):
    '''calculate the var '''
    return np.var(data_array.iloc[:, -1]) * data_array.shape[0]
    pass

def regLeaf(data_array):
    return np.mean(data_array.iloc[:, -1])

def get_best_split(data_array, ops = (1, 4)):
    '''the best point to split data'''
    tols = ops[0]
    toln = ops[1]
    if len(set(data_array.iloc[:, -1])) == 1:
        return None, regLeaf(data_array)
    m, n = data_array.shape
    best_S = np.inf
    best_col = 0
    best_value = 0
    S = getErr(data_array)
    for col in range(n - 1):
        values = set(data_array.iloc[:, col])
        for value in values:
            array_1, array_2 = split_data(data_array, col, value)
            if (array_1.shape[0] < toln) or (array_2.shape[0] < toln):
                continue
            totalError = getErr(array_1) + getErr(array_2)
            if totalError< best_S:
                best_col = col
                best_value = value
                best_S = totalError
    if (S - best_S) < tols:
        return None, regLeaf(data_array)
    array_1, array_2 = split_data(data_array, best_col, best_value)
    if (array_1.shape[0] < toln) or (array_2.shape[0] < toln):
        return None, regLeaf(data_array)

    return best_col, best_value

class node:
    '''tree node'''
    def __init__(self, col=-1, value=None, results=None, gb=None, lb=None):
        self.col = col
        self.value = value
        self.results = results
        self.gb = gb
        self.lb = lb
        pass

def buildTree(data_array, ops = (1, 4)):
    col, val = get_best_split(data_array, ops)
    if col == None:
        return node(results=val)
    else:
        array_1, array_2 = split_data(data_array, col, val)
        greater_branch = buildTree(array_1, ops)
        less_branch = buildTree(array_2, ops)
        return node(col=col, value=val, gb=greater_branch, lb=less_branch)
    pass

def treeCast(tree, inData):
    '''get the classification'''
    if tree.results != None:
        return tree.results
    if inData.iloc[tree.col] > tree.value:
        return treeCast(tree.gb, inData)
    else:
        return treeCast(tree.lb, inData)
    pass

def createForeCast(tree, testData):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeCast(tree, testData.iloc[i])
    return yHat

def GBDT_model(data_array, num_iter, ops = (1, 4)):
    m, n = data_array.shape
    x = data_array.iloc[:, 0:-1]
    y = data_array.iloc[:, -1]
    y = np.mat(y).T
    list_trees = []
    yHat = None
    for i in range(num_iter):
        print('the ', i, ' tree')
        if i == 0:
            tree = buildTree(data_array, ops)
            list_trees.append(tree)
            yHat = createForeCast(tree, x)
        else:
            r = y - yHat
            data_array = np.hstack((x, r))
            data_array = pd.DataFrame(data_array)
            tree = buildTree(data_array, ops)
            list_trees.append(tree)
            rHat = createForeCast(tree, x)
            yHat = yHat + rHat
    return list_trees, yHat

def getwidth(tree):
    if tree.gb == None and tree.lb == None: return 1
    return getwidth(tree.gb) + getwidth(tree.lb)


def getdepth(tree):
    if tree.gb == None and tree.lb == None: return 0
    return max(getdepth(tree.gb), getdepth(tree.lb)) + 1


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if tree.results == None:
        # Get the width of each branch
        w1 = getwidth(tree.lb) * 100
        w2 = getwidth(tree.gb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.lb, left + w1 / 2, y + 100)
        drawnode(draw, tree.gb, right - w2 / 2, y + 100)
    else:
        txt = str(tree.results)
        draw.text((x - 20, y), txt, (0, 0, 0))

if __name__ == '__main__':
    data = loadDataSet('../Data/LogiReg_data.txt')
    tree = buildTree(data)
    drawtree(tree, jpeg='treeview_cart.jpg')
    gbdt_results, y = GBDT_model(data, 10)
    print(y)
    for i in range(len(y)):
        if y[i] > 0:
            print('1')
        elif y[i] < 0:
            print('0')

