from abc import ABC, abstractmethod # abstract classes
import numpy as np # for fitness function
import random

INT_ARRAY_SIZE = 1000

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

    # crossover and mutation helper method declarations can be added here

# implementations of Node
class BinaryOperator(Node):
    def __init__(self, operator, left_child, right_child):
        self.operator = operator
        self.right_child = right_child
        self.left_child = left_child

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


class UnaryOperator(Node):
    def __init__(self, operator, child):
        self.operator = operator
        self.child = child

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


class Number(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, x):
        return self.value
    
    def __str__(self):
        return f"{self.value:.2f}"
    
    def number_of_nodes(self):
        return 1
    
class X(Node):
    def evaluate(self, x):
        return x
    
    def __str__(self):
        return "x"
    
    def number_of_nodes(self):
        return 1


# probability
def P(p):
    return random.random() < p

def get_random_subtree(t, P_RAND):
    if isinstance(t, BinaryOperator):
        if P(P_RAND):
            return t
        if P(0.5):
            return get_random_subtree(t.left_child, P_RAND * 1.1)
        return get_random_subtree(t.right_child, P_RAND * 1.1)
    elif isinstance(t, Number) or isinstance(t, X):
        return t
    raise NotImplementedError()

def insert_random_subtree(t, s, R_INS):
    if isinstance(t, BinaryOperator):
        if P(R_INS):
            return s
        if P(0.5):
            return insert_random_subtree(t.left_child, s, R_INS * 1.1)
        return insert_random_subtree(t.right_child, s, R_INS * 1.1)
    elif isinstance(t, Number) or isinstance(t, X):
        return s
    raise NotImplementedError()


def crossover_tree(t1, t2):
    P_RAND = 0.1
    P_INS = 0.1
    s1 = get_random_subtree(t1, P_RAND)
    s2 = get_random_subtree(t2, P_RAND)
    return insert_random_subtree(t1, s2, P_INS), insert_random_subtree(t2, s1, P_INS)


def mutate_tree(t):
    P_BINARY_MUT = 0.1
    P_UNARY_MUT = 0.01
    P_NUMBER_MUT = 0.1
    P_SWITCH = 0.1
    P_ADD = 0.7

    if isinstance(t, BinaryOperator):
        # Don't mutate
        if not P(P_BINARY_MUT):
            return BinaryOperator(
                t.operator,
                mutate_tree(t.left_child),
                mutate_tree(t.right_child)
        )
        operators = ['+','-','*','/','^']
        operator = t.operator
        if P(P_BINARY_MUT):
            operator = random.choice(operators)

        return BinaryOperator(
            operator,
            mutate_tree(t.left_child),
            mutate_tree(t.right_child)
        )
    elif isinstance(t, Number):
        operators = ['+','-','*','/','^']
        # Don't mutate
        if not P(P_NUMBER_MUT):
            return t
        # Change random to x
        if P(P_SWITCH):
            return X()
        if P(P_ADD):
            subtrees = [t, Number(random.randint(-10, 10))]
            random.shuffle(subtrees)
            return BinaryOperator(
                random.choice(operators),
                *subtrees
                )

        if t.value == -10:
            return Number(-9)
        elif t.value == 10:
            return Number(9)
        else:
            return Number(t.value + random.randint(-1, 1))
    elif isinstance(t, X):
        return t
    else:
        raise NotImplementedError()


def generate_random_tree(P_ENDTREE):
    P_X = 0.3
    operators = ['+','-','*','/','^']

    if P(P_ENDTREE):
        if P(P_X):
            return X()
        return Number(random.randint(-10, 10))
    return BinaryOperator(
        random.choice(operators),
        generate_random_tree(P_ENDTREE * 1.1),
        generate_random_tree(P_ENDTREE * 1.1),
    )


# parser
def parsePolishNotationToTree(str):
    def parseTokensToTreePolish(tokens, idx):
        match(tokens[idx]):
            case '+' | '-' | '*' | '/' | '^': 
                operator = tokens[idx]
                idx = idx + 1

                left_child, idx = parseTokensToTreePolish(tokens, idx)
                right_child, idx = parseTokensToTreePolish(tokens, idx)

                return BinaryOperator(operator, left_child, right_child), idx
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
    LONG_EQUATION_PENALTY = 0.0001
    try:
        fitness = -np.sum(np.abs(ys - tree.evaluate(xs)))
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

    # (3 + 5) * x
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
