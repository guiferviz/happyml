

import numpy as np

from happyml.models import Model


class Element(object):

    id_count = 0

    def __init__(self, **args):
        self.name = args.get("name", None)
        self.value = args.get("value", None)
        if self.value is None:
            shape = args.get("shape", ())
            self.value = (np.random.rand(*shape) - 0.5)
        self.value = np.asfarray(self.value)
        self.shape = self.value.shape
        self.inputs = args.get("inputs", [])
        for i in range(len(self.inputs)):
            self.inputs[i] = as_element(self.inputs[i])
        self.is_input = args.get("is_input", False)
        self.is_parameter = args.get("is_parameter", False)
        self.has_parameter = args.get("has_parameter", False) or \
                             any(i.has_parameter for i in self.inputs)

        self.id = Element.id_count
        Element.id_count += 1

    def forward(self):
        pass

    def backward(self, gradients):
        pass

    def get_computation_path(self):
        stack = [(self, 0)]
        result = []
        visited = set()
        while len(stack) > 0:
            x, i = stack.pop()
            if i == len(x.inputs):
                result.append(x)
            else:
                stack.append((x, i + 1))
                if x.inputs[i] not in visited:
                    stack.append((x.inputs[i], 0))
                    visited.add(x.inputs[i])
        return result

    def set_value(self, value):
        self.value = value.reshape(self.shape)

    def to_model(self, **kwargs):
        return ComputationalGraphModel(self, **kwargs)

    def dot(self, o):
        return Dot(self, o)

    def max(self, o):
        return Max(self, o)

    def __mul__(self, o):
        return Prod([self, o])

    def __rmul__(self, o):
        return Prod([o, self])

    def __add__(self, o):
        return Add([self, o])

    def __radd__(self, o):
        return Add([o, self])

    def __sub__(self, o):
        return Sub([self, o])

    def __rsub__(self, o):
        return Sub([o, self])

    def __neg__(self):
        return Neg(self)

    def __pow__(self, o):
        if o == 2:
            return Square(self)
        raise NotImplementedError()
        #return Pow(self, o)

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self


class Input(Element):

    def __init__(self, value=None, **args):
        Element.__init__(self, value=value,
                               is_input=True,
                               **args)

    def __str__(self):
        string = Element.__str__(self)
        if type(string) != str:
            return "Input(%f)" % self.value

        return string


class Parameter(Element):

    def __init__(self, value=None, **args):
        Element.__init__(self, value=value,
                               is_parameter=True,
                               has_parameter=True,
                               **args)

    def __str__(self):
        string = Element.__str__(self)
        if type(string) != str:
            return "Parameter(%f)" % self.value

        return string


class Constant(Element):

    def __init__(self, value, **args):
        Element.__init__(self, value=value, name=str(value), **args)


class Add(Element):

    def __init__(self, inputs, **args):
        res = 1
        for i in inputs: res += i.value
        Element.__init__(self, inputs=inputs,
                               name="+",
                               shape=i.shape,
                               **args)

    def forward(self):
        result = 0
        for i in self.inputs:
            result += i.value

        self.value = result

    def backward(self, gradients):
        for i in self.inputs:
            if i.has_parameter:
                try:
                    gradients[i] += gradients[self]
                except ValueError as e:
                    print i, i.value.shape
                    print gradients[i].shape, gradients[self].shape
                    print gradients[i] + gradients[self]
                    raise e

    def __str__(self):
        return "(%s)" % " + ".join(str(i) for i in self.inputs)


class Sub(Element):

    def __init__(self, inputs, **args):
        res = 1
        for i in inputs: res += i.value
        Element.__init__(self, inputs=inputs,
                               shape=i.shape,
                               name="-",
                               **args)

    def forward(self):
        result = np.array(self.inputs[0].value)
        for i in range(1, len(self.inputs)):
            result -= self.inputs[i].value

        self.value = result

    def backward(self, gradients):
        grad = 1
        for i in self.inputs:
            if i.has_parameter:
                gradients[i] += grad * gradients[self]
            grad = -1

    def __str__(self):
        return "(%s)" % " - ".join(str(i) for i in self.inputs)


class Neg(Element):

    def __init__(self, input, **args):
        Element.__init__(self, inputs=[input], name="-", **args)

    def forward(self):
        self.value = self.inputs[0].value

    def backward(self, gradients):
        i = self.inputs[0]
        if i.has_parameter:
            gradients[i] += -np.ones(i.value.shape)

    def __str__(self):
        return "(-%s)" % str(self.inputs[0])


class Prod(Element):

    def __init__(self, inputs, **args):
        Element.__init__(self, inputs=inputs, name="*", **args)

    def forward(self):
        result = 1
        for i in self.inputs:
            result *= i.value
        self.value = result

    def backward(self, gradients):
        val = 1
        for i in self.inputs:
            val *= i.value
        
        n = len(self.inputs)
        for i in range(n):
            element = self.inputs[i]
            if element.has_parameter:
                value = 1
                for j in range(n):
                    if j == i: continue
                    value *= self.inputs[j].value
                gradients[element] += value * gradients[self]

    def __str__(self):
        return "(%s)" % " * ".join(str(i) for i in self.inputs)


class Dot(Element):

    def __init__(self, input1, input2, **args):
        shape = np.dot(input1.value, input2.value).shape
        Element.__init__(self, inputs=[input1, input2],
                               name=".",
                               shape=shape,
                               **args)

    def forward(self):
        self.value = np.dot(self.inputs[0].value, self.inputs[1].value)

    def _gradient(self, a, b, g):
        if len(b.shape) > 1:
            g[a] += np.dot(g[self], b.value)
        elif len(a.shape) > 1:
            g[a] += np.einsum('i,j->ij', g[self], b.value)
        else:
            g[a] += np.dot(g[self], b.value)

    def backward(self, gradients):
        a, b = self.inputs
        if a.has_parameter:
            self._gradient(a, b, gradients)
        if b.has_parameter:
            self._gradient(b, a, gradients)

    def __str__(self):
        return "(%s)" % " . ".join(str(i) for i in self.inputs)


class Square(Element):

    def __init__(self, element, **args):
        Element.__init__(self, inputs=[element],
                               shape=element.shape,
                               name="^2",
                               **args)

    def forward(self):
        self.value = np.square(self.inputs[0].value)

    def backward(self, gradients):
        element = self.inputs[0]
        if element.has_parameter:
            gradients[element] += 2 * element.value * gradients[self]

    def __str__(self):
        return "(%s ^ 2)" % str(self.inputs[0])


class Abs(Element):

    def __init__(self, element, **args):
        Element.__init__(self, inputs=[element,],
                               name="abs",
                               **args)

    def forward(self):
        self.value = np.abs(self.inputs[0].value)

    def backward(self, gradients):
        element = self.inputs[0]
        if element.has_parameter:
            grad_abs = element.value / self.value
            gradients[element] += grad_abs * gradients[self]

    def __str__(self):
        return "abs(%s)" % str(self.inputs[0])



class Max(Element):

    def __init__(self, input1, input2, **args):
        input1 = as_element(input1)
        input2 = as_element(input2)
        Element.__init__(self, inputs=[input1, input2],
                               name="max",
                               **args)

    def forward(self):
        self.value = np.maximum(self.inputs[0].value,
                                self.inputs[1].value)

    def backward(self, gradients):
        input1, input2 = self.inputs
        if input1.has_parameter:
            gradients[input1] += (input1.value >= input2.value).astype(
                float) * gradients[self]
        if input2.has_parameter:
            gradients[input2] += (input2.value > input1.value).astype(
                float) * gradients[self]

    def __str__(self):
        return "max(%s, %s)" % (str(self.inputs[0]), str(self.inputs[1]))


class ReduceSum(Element):

    def __init__(self, element, **args):
        shape = np.sum(element.value).shape
        Element.__init__(self, inputs=[element],
                               name="sum",
                               shape=shape,
                               **args)

    def forward(self):
        self.value = np.sum(self.inputs[0].value)

    def backward(self, gradients):
        element = self.inputs[0]
        if element.has_parameter:
            gradients[element] += gradients[self]

    def __str__(self):
        return "sum%s" % str(self.inputs[0])


class Tanh(Element):

    def __init__(self, element, **args):
        args.setdefault("name", "tanh")
        Element.__init__(self, inputs=[element],
                               shape=element.shape,
                               **args)

    def forward(self):
        self.value = np.tanh(self.inputs[0].value)

    def backward(self, gradients):
        element = self.inputs[0]
        if element.has_parameter:
            gradients[element] += (1 - np.square(self.value)) * gradients[self]


class Sigmoid(Element):

    def __init__(self, element, **args):
        args.setdefault("name", "sigmoid")
        Element.__init__(self, inputs=[element],
                               shape=element.shape,
                               **args)

    def forward(self):
        element = self.inputs[0]
        self.value = 1 / (1 + np.exp(-element.value))

    def backward(self, gradients):
        element = self.inputs[0]
        if element.has_parameter:
            gradients[element] += self.value * (1 - self.value) * gradients[self]


class ReLU(Element):

    def __init__(self, element, epsilon=0, **args):
        self.epsilon = epsilon
        args.setdefault("name", "ReLU")
        Element.__init__(self, inputs=[element],
                               shape=element.shape,
                               **args)

    def forward(self):
        element = self.inputs[0]
        self.value = np.maximum(self.epsilon, element.value)

    def backward(self, gradients):
        element = self.inputs[0]
        if element.has_parameter:
            grad_max = (element.value >= self.epsilon).astype(float)
            gradients[element] += grad_max * gradients[self]


class ComputationalGraphModel(Model):

    def __init__(self, graph, out_shape=None):
        inputs = [i for i in graph.get_computation_path() if i.is_input]
        # FIXME: computational graphs accepts only one input.
        self.x = inputs[0]
        self.out_shape = out_shape or (1,)
        self.graph = graph

    def h(self, x):
        self.x.set_value(x)
        return forward_all(self.graph).reshape(self.out_shape)


def as_element(x):
    if isinstance(x, Element):
        return x
    elif isinstance(x, (int, long, float)):
        return Constant(x)

    raise ValueError("Unknow conversion to Element: %s" % type(x))


def forward_all(x):
    path = x.get_computation_path()
    for i in path:
        i.forward()
    return x.value


def backward_all(x, gradients=None):
    path = x.get_computation_path()
    gradients = gradients if gradients is not None else {}
    gradients.setdefault(x, np.ones(x.shape))

    for i in path:
        if i.has_parameter:
            gradients.setdefault(i, np.zeros(i.shape))

    for i in reversed(path):
        if not i.has_parameter:
            continue

        """ Check gradients and values has the same shape.
        print "-----------", i
        for j in path:
            if j.has_parameter and j in gradients:
                print j
                print "\t", j.shape, gradients[j].shape
                if j.shape != gradients[j].shape:
                    raise ValueError("Shape gradient and value not the same")
        """
        i.backward(gradients)

        if not i.is_parameter:
            del gradients[i]

    return gradients


def check_gradients(x, delta=1e-6, rtol=0.01, verbose=False):
    params = [i for i in x.get_computation_path() if i.is_parameter]
    forward_all(x)
    gradients = backward_all(x)

    for param in params:
        g = np.zeros(param.shape)
        for i in np.ndindex(param.shape):
            v = param.value[i]
            param.value[i] = v - delta
            val_minus = forward_all(x)
            param.value[i] = v + delta
            val_plus = forward_all(x)
            param.value[i] = v
            g[i] = (val_plus - val_minus) / (2 * delta)
        match = np.allclose(g, gradients[param], rtol=rtol)
        if verbose or not match:
            print "----------------------------------------------------"
            print "param: %s" % str(param)
            print "gradient (finite difference): %s" % g
            print "gradient (automatic): %s" % gradients[param]
        if not match:
            raise ValueError("gradients do not match")
