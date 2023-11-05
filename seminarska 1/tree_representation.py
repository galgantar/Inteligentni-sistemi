from abc import ABC, abstractmethod # abstract classes
import numpy as np # for fitness function

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

    # evaluate at x = 5
    print(test.evaluate(5))
    print(shouldBeSameAs.evaluate(5))

    # test fitness function
    xs = np.array([1, 2, 3])
    ys = np.array([1, 4, 9])
    true_expression = parsePolishNotationToTree("^ x 2")
    print(fitness(test, xs, ys))
    print(fitness(true_expression, xs, ys))
