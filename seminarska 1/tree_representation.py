from abc import ABC, abstractmethod # abstract classes
import random


# base class
class Node(ABC):
    @abstractmethod
    def evaluate(self, x):
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


class UnaryOperator(Node):
    def __init__(self, operator, child):
        self.operator = operator
        self.child = child

    def evaluate(self, x):
        match(self.operator):
            case _: # default case
                raise NotImplementedError()


class Number(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, x):
        return self.value
    
class X(Node):
    def evaluate(self, x):
        return x


def mutate_tree(t):
    P_BINARY_MUT = 0.01
    P_UNARY_MUT = 0.01
    P_NUMBER_MUT = 0.01
    P_SWITCH = 0.001
    
    if isinstance(t, BinaryOperator):
        # Don't mutate
        if random.random() >= P_BINARY_MUT:
            return BinaryOperator(
                t.operator,
                mutate_tree(t.left_child),
                t.right_child(t.right_child)
        )
        operators = ['+','-','*','/','*']
        return BinaryOperator(
            random.choice(operators),
            mutate_tree(t.left_child),
            mutate_tree(t.right_child)
        )

    elif isinstance(t, Number):
        # Don't mutate
        if random.random() >= P_NUMBER_MUT:
            return t
        # Change random to x
        if random.random() < P_SWITCH:
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
            case '+' | '-' | '*' | '/' | '*': 
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


# main (for testing purposes)
if __name__ == "__main__":

    # (3 + 5) * x
    test = parsePolishNotationToTree("* + 3 5 x")

    print("MUTATION", mutate_tree(test))

    # Our tree does not implement __eq__ and consequently cannot be compared.
    shouldBeSameAs = BinaryOperator('*', BinaryOperator('+', Number(3.0), Number(5.0)), X())

    # evaluate at x = 5
    print(test.evaluate(5))
    print(shouldBeSameAs.evaluate(5))