from abc import ABC, abstractmethod # abstract classes
import numpy as np # for fitness function
import random


# base class
class Node(ABC):
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def __str__(self):
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
                return self.left_child.evaluate(x) + self.right_child.evaluate(x)
            case '*':
                return self.left_child.evaluate(x) * self.right_child.evaluate(x)
            case '/':
                return self.left_child.evaluate(x) / self.right_child.evaluate(x)
            case '^':
                return self.left_child.evaluate(x) ** self.right_child.evaluate(x)
            case _: # default case
                raise NotImplementedError()
            
    def __str__(self):
        return f"({self.operator} {self.left_child} {self.right_child})"


class UnaryOperator(Node):
    def __init__(self, operator, child):
        self.operator = operator
        self.child = child

    def evaluate(self, x):
        match(self.operator):
            case _: # default case
                raise NotImplementedError()
            
    def __str__(self):
        return f"({self.operator} {self.child})"


class Number(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, x):
        return self.value
    
    def __str__(self):
        return f"{self.value}"
    
class X(Node):
    def evaluate(self, x):
        return x
    
    def __str__(self):
        return "x"


# probability
def P(p):
    return random.random() < p

def get_random_subtree(t):
    P_RAND = 0.1
    if isinstance(t, BinaryOperator):
        if P(P_RAND):
            return t
        if P(0.5):
            return get_random_subtree(t.left_child)
        return get_random_subtree(t.right_child)
    elif isinstance(t, Number) or isinstance(t, X):
        return t
    raise NotImplementedError()

def insert_random_subtree(t, s):
    R_INS = 0.1
    if isinstance(t, BinaryOperator):
        if P(R_INS):
            return s
        if P(0.5):
            return insert_random_subtree(t.left_child, s)
        return insert_random_subtree(t.right_child, s)
    elif isinstance(t, Number) or isinstance(s, X):
        return s
    raise NotImplementedError()


def crossover_tree(t1, t2):
    s1 = get_random_subtree(t1)
    s2 = get_random_subtree(t2)
    return insert_random_subtree(t1, s2), insert_random_subtree(t2, s1)


def mutate_tree(t):
    P_BINARY_MUT = 0.01
    P_UNARY_MUT = 0.01
    P_NUMBER_MUT = 0.01
    P_SWITCH = 0.001
    
    if isinstance(t, BinaryOperator):
        # Don't mutate
        if not P(P_BINARY_MUT):
            return BinaryOperator(
                t.operator,
                mutate_tree(t.left_child),
                mutate_tree(t.right_child)
        )
        operators = ['+','-','*','/','*']
        return BinaryOperator(
            random.choice(operators),
            mutate_tree(t.left_child),
            mutate_tree(t.right_child)
        )
    elif isinstance(t, Number):
        # Don't mutate
        if not P(P_NUMBER_MUT):
            return t
        # Change random to x
        if P(P_SWITCH):
            return X()
        return Number(random.uniform(-100, 100))
    elif isinstance(t, X):
        return t
    else:
        raise NotImplementedError()


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
    return -np.sum(np.square(ys - tree.evaluate(xs)))


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
