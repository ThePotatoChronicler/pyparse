"""
A really bad math parser.
Last Update: 14. May, 2021



To anyone reading this:
Do not thread any further, for all
that awaits is pain and suffering.
The code ahead is an example of
what you should strive not
to write. The code uses improper
workarounds, lacks comments,
makes questionable design
decisions and more. Please, do
not take example from whatever
is written here, except as a bad
example, because that's all this
is. - Potato Chronicler
"""

from abc import ABC, abstractmethod

class MathObject(ABC):
    """
    Abstact Math object class

    All objects which can be parsed and compiled
    are meant to be subclasses of this class.
    """
    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError()

class MathOp(ABC):
    """
    Abstract Math operation class

    All operations are meant to be
    subclasses of this class, the default
    parser creates them as such, and
    the default compiler expects them as
    subclasses of this class.
    """
    def __init__(self, l: int = 1, r:int = 1):
        """
        l = How many tokens/values to get from left
        r = How many tokens/values to get from right
        """
        self.l = l
        self.r = r

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError()

class MathValue(MathObject):
    """
    Default Math value class

    Holds a value meant to be parsed and created
    by the default parser/compiler.
    """
    def __init__(self, val):
        self._val = val

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)

    @property
    def value(self):
        return self._val

class MathLib(ABC):
    """
    Abstract Math function library class

    Math libraries by default are
    expected to be callable, returning a dictionary
    of Math functions.
    """
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()

class MathFunc(MathValue):
    """
    Abstract Math function class

    Math functions by default are
    expected to have the mi(n) and ma(x) properties,
    which tells the parser how many arguments it should take,
    preventing duplicate code to handle inproper amount of arguments.

    The default parser will assume that when ma(x) is negative,
    the value will be ignored and any number of arguments above
    mi(n) will be accepted.

    Currently, mi(n) being negative has no meaning.

    By default, functions should expect to receive resolved values,
    not wrapped in MathObject or MathValue. As such is the
    behaviour of the default parser and compiler.
    """

    def __init__(self, mi : int = 1, ma : int = 1, *args):
        self.args = args
        self.mi = mi
        self.ma = ma

class CommonOps:
    """
    Common operations used by the default parser/compiler.
    """
    class Add(MathOp):
        def __call__(self, l, r):
            if len(l) == 1:
                return l[0] + r[0]
            else:
                return r[0]

        def __repr__(self):
            return "'+'"

        def __str__(self):
            return '+'

    class Sub(MathOp):
        def __call__(self, l, r):
            if len(l) == 1:
                return l[0] - r[0]
            else:
                return -r[0]

        def __repr__(self):
            return "'-'"

        def __str__(self):
            return '-'

    class Mult(MathOp):
        def __call__(self, l, r):
            return l[0] * r[0]

        def __repr__(self):
            return "'*'"

        def __str__(self):
            return '*'

    class Div(MathOp):
        def __call__(self, l, r):
            return l[0] / r[0]

        def __repr__(self):
            return "'/'"

        def __str__(self):
            return '/'

class CommonFunc(MathLib):
    class sin(MathFunc):
        @property
        def value(self):
            from math import sin
            return sin(self.args[0])

    def __call__(self):
        return {
            'sin' : self.sin,
        }



class Math(MathValue):
    """
    The default parser/compiler
    """

    class Func(MathObject):
        """
        A function wrapper.

        Meant to be wrapped around MathFunc objects
        used by the default parser.
        It's in it's own special class so that the compiler
        can detect it and resolve the function and arguments it holds.
        """
        def __init__(self, m, t):
            self.m = m
            self.t = t

        @property
        def value(self):
            return self.m.funcdict[self.t[0]](*tuple(v.value for v in self.t[1])).value

    def __init__(self, source:str = "0", init_parse:bool = True, init_compile:bool = True, math_libs:tuple = (CommonFunc(),)):
        self.source = source

        self._val = None
        self.tokens = None
        self.funcdict = {}
        for o in math_libs:
            self.funcdict.update(o())

        if init_parse:
            self.tokens = self.parse()

        if init_parse and init_compile:
            self._val = self.compile()

    @property
    def value(self):
        if self._val == None:
            if self.tokens == None: self.tokens = self.parse()
            self._val = self.compile()
        return self._val

    def parse(self):
        import re

        sourceb = self.source

        # Pulls functions out of the source,
        # slapping them into an array

        funcobjs = []
        while True:
            match = re.search(r" *([_a-zA-Z][_a-zA-Z0-9]*) *\(", sourceb)
            if not match: break

            span = match.span(0)
            point = span[1]
            depth = 1
            seps = [point]
            while depth:
                if sourceb[point] == ',' and depth == 1:
                    seps.append(point)
                    seps.append(point + 1)
                if   sourceb[point] == '(':
                    depth += 1
                elif sourceb[point] == ')' :
                    depth -= 1
                point += 1
            seps.append(point - 1)
            args = tuple(Math(sourceb[seps[i]:seps[i+1]], False, False) for i in range(0, len(seps) - 1, 2))
            funcobjs.append((match.group(1), args))

            # Update source, removing the matched function part
            sourceb = sourceb[:span[0]] + '[f]' + sourceb[point:]

        # Finds numbers and operators, adds spaces around them
        # If this is not done, the tokenizing part doesn't work
        def subnums(m):
            return ' ' + m.group(0).strip() + ' '
        sourceb = re.sub(r"( *\d+\.?\d* *)", subnums, sourceb)
        sourceb = re.sub(r"( *[-+/*] *)", subnums, sourceb)

        # Adds brackets around /
        # because operator precedence
        if not re.fullmatch(r"^[( ]*(\d+\.?\d* / \d+\.?\d*)[) ]*$", sourceb):
            def divfunc(m):
                return ' (' + m.group(0) + ') '
            sourceb = re.sub(r" *(\d+\.?\d* / \d+\.?\d*) *", divfunc, sourceb)

        # Finding all brackets and replacing them,
        # this should help with making less of a mess
        bracketobs = []
        point = 0
        point2 = 0
        depth = 0
        while point < len(sourceb):
            if sourceb[point] == '(':
                depth = 1
                point2 = point + 1
                while depth:
                    if   sourceb[point2] == '(':
                        depth += 1
                    elif sourceb[point2] == ')' :
                        depth -= 1
                    point2 += 1
                    if point2 > len(sourceb):
                        raise SyntaxError("Unclosed bracket!")
                point2 -= 1

                bracketobs.append(Math(sourceb[point + 1 : point2], False, False))
                sourceb = sourceb[0 : point + 1] + sourceb[point2:]
            point += 1

        # Actually making them tokens
        tokens = []
        brackiter = iter(bracketobs)
        funciter = iter(funcobjs)
        for token in sourceb.split():

            # Functions
            if token == "[f]":
                funct = next(funciter)
                argl = len(funct[1])
                tempinst = self.funcdict[funct[0]]
                mi = tempinst.mi
                ma = tempinst.ma
                if argl < mi or (argl > ma or ma < 0):
                    raise ValueError(f"Math function {tempinst} received wrong amount of arguments ({mi} <= {argl} <= {ma}).")
                tokens.append(self.Func(self, funct))
                continue

            # Numbers
            try:
                _f = float(token)
            except ValueError:
                pass
            else:
                tokens.append(MathValue(_f))
                continue

            # Brackets
            if token == "()":
                tokens.append(next(brackiter))
                continue

            # Addition
            if token == "+":
                tokens.append(CommonOps.Add())
                continue
            # Substraction
            if token == "-":
                tokens.append(CommonOps.Sub())
                continue
            # Multiplication
            if token == "*":
                tokens.append(CommonOps.Mult())
                continue
            # Division
            if token == "/":
                tokens.append(CommonOps.Div())
                continue

            # Last option
            tokens.append(MathValue(token))

        return tokens

    def compile(self):
        stack = []
        value = 0
        point = 0
        tlen = len(self.tokens)

        while point < tlen:
            p = self.tokens[point]

            if isinstance(p, MathObject):
                stack.append(p)
                point += 1
                continue

            if isinstance(p, MathOp):
                point += 1
                argsl = []
                try:
                    for _ in range(p.l):
                        argsl.append(stack.pop(0).value)
                except IndexError:
                    pass
                argsl = tuple(argsl)
                argsr = tuple(v.value for v in self.tokens[point : point + p.r])
                point += len(argsr)
                result = p(argsl, argsr)
                if not isinstance(result, MathObject):
                    stack.append(MathValue(result))
                else:
                    stack.append(result)
                continue

            raise ValueError(f"Unknown value in tokens at index {point}", self.tokens)

        if len(stack) == 1:
            value = stack[0].value
        else:
            stack = list(map(lambda x: x.value, stack))
            from functools import reduce
            value = reduce(lambda x, y: x + y, stack)
            return value

        try:
            return int(value) if value == int(value) else value
        except ValueError:
            return value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source})"

    def __str__(self):
        return str(self.value)

print(Math("(1 + 1) +3 + 2 + (5 + 2)"))
print(Math("1 + 2 + 3 + (2 + 3 + (2 + 8 + (2 + 1)))"))
print(Math("5 - (-2)"))
print(Math("5 * (-5)"))
print(Math("64 / 8 + 1"))
print(Math("5 - 1 / 2"))
print(Math("2 * (4 / 2)"))
print(Math("2 * ((4 / 2))"))
print(Math("It doesn't like strings"))

print(Math("2 2"))
# Example of what happens when
# you leave multiple values on the stack
# when the compiler goes through all
# available tokens.

print(Math("5 +sin(sin(50) + 2)-sin(35)"))
