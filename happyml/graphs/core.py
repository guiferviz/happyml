

import numpy as np

from happyml.models import Hypothesis


class Element(object):

    id_count = 0

    def __init__(self, **args):
        shape = args.get("shape", ())
        self.value = args.get("value", None)
        if self.value is None:
            self.value = np.random.rand(*shape)
        self.value = np.asfarray(self.value)
        self.shape = self.value.shape
        self.inputs = args.get("inputs", [])
        self.name = args.get("name", None)
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
                stack.append((x, i+1))
                if x.inputs[i] not in visited:
                    stack.append((x.inputs[i], 0))
                    visited.add(x.inputs[i])
        return result

    def set_value(self, value):
        self.value = value.reshape(self.shape)

    def to_model(self):
        return ComputationalGraphModel(self)

	def dot(self, o):
		return Dot(self, o)

    def __mul__(self, o):
        return Prod([self, o])

    def __add__(self, o):
        return Add([self, o])

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


class Add(Element):

    def __init__(self, inputs, **args):
        Element.__init__(self, inputs=inputs, name="+", **args)

    def forward(self):
        result = 0
        for i in self.inputs:
            result += i.value

        self.value = result

    def backward(self, gradients):
        for i in self.inputs:
            if i.has_parameter:
                gradients[i] += gradients[self]

    def __str__(self):
        return " + ".join(str(i) for i in self.inputs)


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
        
        for i in self.inputs:
            if i.has_parameter:
                gradients[i] += val / i.value * gradients[self]

    def __str__(self):
        return " * ".join(str(i) for i in self.inputs)


class Dot(Element):

    def __init__(self, input1, input2, **args):
        Element.__init__(self, inputs=[input1, input2],
        					   name=".",
        					   **args)

    def forward(self):
        self.value = np.dot(self.inputs[0].value, self.inputs[1].value)

    def backward(self, gradients):
        if self.inputs[0].has_parameter:
            gradients[self.inputs[0]] += self.inputs[1].value * \
            							 gradients[self]
        if self.inputs[1].has_parameter:
            gradients[self.inputs[1]] += self.inputs[0].value * \
            							 gradients[self]

    def __str__(self):
        return " . ".join(str(i) for i in self.inputs)


class ComputationalGraphModel(Hypothesis):

    def __init__(self, graph):
        inputs = [i for i in graph.get_computation_path() if i.is_input]
        # FIXME: computational graphs accepts only one input.
        self.x = inputs[0]
        self.graph = graph

    def h(self, x):
        self.x.set_value(x)
        return forward_all(self.graph).reshape((1,))


def forward_all(x):
    path = x.get_computation_path()
    for i in path:
        i.forward()
    return x.value


def backward_all(x, loss_gradient=1):
    path = x.get_computation_path()
    gradients = {x: loss_gradient}
    for i in path:
        if i.has_parameter:
            gradients.setdefault(i, np.zeros(i.value.shape))

    for i in reversed(path):
        if not i.has_parameter:
            continue

        i.backward(gradients)

        if not i.is_parameter:
            del gradients[i]

    return gradients
