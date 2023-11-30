from abc import ABC, abstractmethod # abstract classes
import numpy as np # for fitness function
import random

INT_ARRAY_SIZE = 1000
P_GENERATION_X = 0.1
P_ENDTREE_INIT = 0.3
LONG_EQUATION_PENALTY = 0.01


# base class
class Node(ABC):
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def number_of_nodes(self):
        pass
    
    @abstractmethod
    def list_of_nodes(self):
        pass



# implementations of Node
class BinaryOperator(Node):
    def __init__(self, operator, left_child, right_child):
        self.parent = None
        self.operator = operator
        self.right_child = right_child
        self.left_child = left_child
        left_child.parent = self
        right_child.parent = self

    def evaluate(self, x):
        match(self.operator):
            case '+':
                return self.left_child.evaluate(x) + self.right_child.evaluate(x)
            case '-':
                return self.left_child.evaluate(x) - self.right_child.evaluate(x)
            case '*':
                return self.left_child.evaluate(x) * self.right_child.evaluate(x)
            case '/':
                return self.left_child.evaluate(x) / self.right_child.evaluate(x)
            case '^':
                return self.left_child.evaluate(x) ** self.right_child.evaluate(x)
            case 'max':
                return np.maximum(self.left_child.evaluate(x), self.right_child.evaluate(x))
            case 'min':
                return np.minimum(self.left_child.evaluate(x), self.right_child.evaluate(x))
            case _: # default case
                raise NotImplementedError()
            
    def __str__(self):
        return f"{self.operator} {self.left_child} {self.right_child}"
    
    def number_of_nodes(self):
        return 1 + self.left_child.number_of_nodes() + self.right_child.number_of_nodes()
    
    def list_of_nodes(self):
        return [self] + self.left_child.list_of_nodes() + self.right_child.list_of_nodes()


class UnaryOperator(Node):
    def __init__(self, operator, child):
        self.parent = None
        self.operator = operator
        self.child = child
        child.parent = self

    def evaluate(self, x):
        match(self.operator):
            case 'sin':
                return np.sin(self.child.evaluate(x))
            case 'cos':
                return np.cos(self.child.evaluate(x))
            case 'exp':
                return np.exp(self.child.evaluate(x))
            case 'log':
                return np.log(self.child.evaluate(x))
            case 'sqrt':
                return np.sqrt(self.child.evaluate(x))
            case 'abs':
                return np.abs(self.child.evaluate(x))
            case 'neg':
                return -self.child.evaluate(x)
            case _: # default case
                raise NotImplementedError()
            
    def __str__(self):
        return f"{self.operator} {self.child}"
    
    def number_of_nodes(self):
        return 1 + self.child.number_of_nodes()
    
    def list_of_nodes(self):
        return [self] + self.child.list_of_nodes()



class Number(Node):
    def __init__(self, value):
        self.parent = None
        self.value = value

    def evaluate(self, x):
        return self.value
    
    def __str__(self):
        return f"{self.value:.2f}"
    
    def number_of_nodes(self):
        return 1
    
    def list_of_nodes(self):
        return [self]



class X(Node):
    def __init__(self):
        self.parent = None
    
    def evaluate(self, x):
        return x
    
    def __str__(self):
        return "x"
    
    def number_of_nodes(self):
        return 1
    
    def list_of_nodes(self):
        return [self]



# probability
def P(p):
    return random.random() < p

def get_random_subtree(t):
    l = t.list_of_nodes()
    return random.choice(l)
    
def switch(s1, s2):
    p1 = s1.parent
    p2 = s2.parent
    
    if p1 is None or p2 is None:
        return
    
    if p1 is BinaryOperator:
        if p1.left_child == s1:
            p1.left_child = s2
        else:
            p1.right_child = s2
        
        if p2.left_child == s2:
            p2.left_child = s1
        else:
            p2.right_child = s1
    
    elif p1 is UnaryOperator:
        p1.child = s2
        p2.child = s1

    s1.parent = p2
    s2.parent = p1
    

def crossover_tree(t1, t2):
    s1 = get_random_subtree(t1)
    s2 = get_random_subtree(t2)
    
    switch(s1, s2)
    
    return t1, t2


def mutate_tree(t):
    l = t.list_of_nodes()
    s = random.choice(l)
    
    if isinstance(s, BinaryOperator):
        s.operator = random.choice(['+','-','*','/','^','max','min'])
    elif isinstance(s, UnaryOperator):
        s.operator = random.choice(['sin','cos','exp','log','sqrt','abs','neg'])
    elif isinstance(s, Number):
        s.value = random.randint(-10, 10)
    elif isinstance(s, X):
        pass
    else:
        raise NotImplementedError()
    
    return t


def generate_random_tree(P_ENDTREE = None):
    binary_operators = ['+','-','*','/','^','max','min']
    unary_operators = ['sin','cos','exp','log','sqrt','abs','neg']
    P_ENDTREE = P_ENDTREE_INIT if P_ENDTREE == None else P_ENDTREE

    if P(P_ENDTREE):
        if P(P_GENERATION_X):
            return X()
        return Number(random.randint(-10, 10))

    op = random.choice(binary_operators + unary_operators)
    if op in unary_operators:
        return UnaryOperator(
            op,
            generate_random_tree(P_ENDTREE * 1.1),
        )
    else:
        return BinaryOperator(
            op,
            generate_random_tree(P_ENDTREE * 1.1),
            generate_random_tree(P_ENDTREE * 1.1)
        )


# parser
def parsePolishNotationToTree(str):
    def parseTokensToTreePolish(tokens, idx):
        match(tokens[idx]):
            case '+' | '-' | '*' | '/' | '^' | 'max' | 'min': 
                operator = tokens[idx]
                idx = idx + 1

                left_child, idx = parseTokensToTreePolish(tokens, idx)
                right_child, idx = parseTokensToTreePolish(tokens, idx)

                return BinaryOperator(operator, left_child, right_child), idx
            case 'sin' | 'cos' | 'exp' | 'log' | 'sqrt' | 'abs' | 'neg':
                operator = tokens[idx]
                idx = idx + 1

                child, idx = parseTokensToTreePolish(tokens, idx)

                return UnaryOperator(operator, child), idx
            case 'x':
                # x = tokens[idx]
                idx = idx + 1
                return X(), idx
            case _:
                number = float(tokens[idx])
                idx = idx + 1
                return Number(number), idx
            
    tokens = str.split(' ')
    tree, _ = parseTokensToTreePolish(tokens, 0)
    return tree

# fitness function
def fitness(tree, xs, ys):
    try:
        fitness = -np.sum(np.square(ys - tree.evaluate(xs)))
    except:
        fitness = -np.inf
    
    if np.isfinite(fitness) and not np.iscomplexobj(fitness):
        return fitness * (1 + LONG_EQUATION_PENALTY * tree.number_of_nodes())
    else:
        return -np.inf

def toIntArray(tree):
    bytes = str(tree).encode('utf-8')
    arr = np.zeros(INT_ARRAY_SIZE, dtype=np.int8)
    for i in range(len(bytes)):
        arr[i] = bytes[i]
    return arr

def fromIntArray(arr):
    return parsePolishNotationToTree(arr.tobytes().decode('utf-8').replace('\x00', ''))

# main (for testing purposes)
if __name__ == "__main__":
    test = parsePolishNotationToTree("* + 3 5 x")
    shouldBeSameAs = BinaryOperator('*', BinaryOperator('+', Number(3.0), Number(5.0)), X())
    print(test)
    print(shouldBeSameAs)

    print("MUTATION", mutate_tree(test))

    # evaluate at x = 5
    print(test.evaluate(5))
    print(shouldBeSameAs.evaluate(5))

    # test fitness function
    xs = np.array([1, 2, 3])
    ys = np.array([1, 4, 9])
    true_expression = parsePolishNotationToTree("^ x 2")
    print(fitness(test, xs, ys))
    print(fitness(true_expression, xs, ys))

    for i in range(5):
        t1 = generate_random_tree(0.3)
        t2 = generate_random_tree(0.3)
        print("T1", t1.evaluate(5))
        print("T2", t2.evaluate(5))
        t1, t2 = crossover_tree(t1, t2)
        print("T1 NEW", t1.evaluate(5))
        print("T2 NEW", t2.evaluate(5))

    print("int array: ", toIntArray(test))
    print("from int array: ", fromIntArray(toIntArray(test)))

    test = parsePolishNotationToTree("* + x 5 6")
    print(test.contains_x(), test.evaluate(12))
    test.simplify()
    print(test, test.evaluate(12))
    
    for i in range(500):
        test = generate_random_tree(0.3)
        print(test)
        print(test.evaluate(5))

