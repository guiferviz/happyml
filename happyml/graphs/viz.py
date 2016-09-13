

from graphviz import Digraph


def graph2dot(x, **kwargs):
    dot = Digraph(body=["rankdir=LR;"], **kwargs)

    path = x.get_computation_path()
    for i in path:
        if i.is_input:
            dot.node(str(i.id), i.name, color="green")
        elif i.is_parameter:
            dot.node(str(i.id), i.name, color="gold")
        else:
            dot.node(str(i.id), i.name)
        
        for ii in i.inputs:
            dot.edge(str(ii.id), str(i.id))

    return dot
