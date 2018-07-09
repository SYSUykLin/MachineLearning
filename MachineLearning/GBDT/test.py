import numpy as np
from PIL import Image, ImageDraw


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def splitData(data_array, col, value):
    array_1 = data_array[data_array[:, col] >= value, :]
    array_2 = data_array[data_array[:, col] < value, :]
    return array_1, array_2


def getErr(data_array):
    return np.var(data_array[:, -1]) * data_array.shape[0]


def regLeaf(data_array):  # returns the value used for each leaf
    return np.mean(data_array[:, -1])


def get_best_split(data_array, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(data_array[:, -1])) == 1:
        return None, regLeaf(data_array)
    m, n = data_array.shape
    best_S = np.inf
    best_col = 0
    best_value = 0
    S = getErr(data_array)
    for col in range(n - 1):
        values = set(data_array[:, col])
        for value in values:
            array_1, array_2 = splitData(data_array, col, value)
            if (array_1.shape[0] < tolN) or (array_2.shape[0] < tolN):
                continue
            total_error = getErr(array_1) + getErr(array_2)
            if total_error < best_S:
                best_col = col
                best_value = value
                best_S = total_error
    if (S - best_S) < tolS:
        return None, regLeaf(data_array)

    array_1, array_2 = splitData(data_array, best_col, best_value)
    if (array_1.shape[0] < tolN) or (array_2.shape[0] < tolN):
        return None, regLeaf(data_array)

    return best_col, best_value


class node:
    def __init__(self, col=-1, value=None, results=None, gb=None, lb=None):
        self.col = col
        self.value = value
        self.results = results
        self.gb = gb
        self.lb = lb


def buildTree(data_array, ops=(1, 4)):
    col, val = get_best_split(data_array, ops)  # choose the best split
    if col == None:
        return node(results=val)
    else:
        array_1, array_2 = splitData(data_array, col, val)
        greater_branch = buildTree(array_1, ops)
        less_branch = buildTree(array_2, ops)
        return node(col=col, value=val, gb=greater_branch, lb=less_branch)


def treeForeCast(tree, inData):
    if tree.results != None:
        return tree.results
    # print 'tree.col:',tree.col
    if inData[tree.col] > tree.value:
        return treeForeCast(tree.gb, inData)
    else:
        return treeForeCast(tree.lb, inData)


def createForeCast(tree, testData):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, testData[i])
    return yHat


def gbdt(data_array, num_iter, ops=(1, 4)):
    m, n = data_array.shape
    x = data_array[:, 0:-1]
    y = data_array[:, -1].reshape((m, 1))
    list_trees = []
    for i in range(num_iter):
        print
        'i: ', i
        if i == 0:
            tree = buildTree(data_array, ops)
            list_trees.append(tree)
            yHat = createForeCast(tree, x)
        else:
            r = y - np.array(yHat)
            data_array = np.hstack((x, r))
            tree = buildTree(data_array, ops)
            list_trees.append(tree)
            rHat = createForeCast(tree, x)
            yHat = yHat + rHat
    return list_trees


def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results != None:
        print
        str(tree.results)
    else:
        # Print the criteria
        print
        str(tree.col) + ':' + str(tree.value) + '? '

        # Print the branches
        print
        indent + 'T->',
        printtree(tree.gb, indent + '  ')
        print
        indent + 'F->',
        printtree(tree.lb, indent + '  ')


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
        # txt = 'i am god'
        draw.text((x - 20, y), txt, (0, 0, 0))


if __name__ == '__main__':
    # trainMat=(loadDataSet('bikeSpeedVsIq_train.txt'))
    # testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    data = loadDataSet('../Data/ex0.txt')
    data_array = np.array(data)
    tree = buildTree(data_array)
    drawtree(tree, jpeg='treeview_cart.jpg')

    gbdt_results = gbdt(data_array, 10)