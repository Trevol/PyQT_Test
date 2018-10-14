from atom.api import Atom, ContainerList

class Mm(Atom):
    l = ContainerList()

m = Mm()
m.l = [1, '2', 5.6]
print(5.6 in m.l)