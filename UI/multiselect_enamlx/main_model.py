from atom.api import Atom, ContainerList, Typed, Instance

class MainModel(Atom):
    items = ContainerList()
    checked_items = ContainerList()
    selected_item = Typed(object)


if __name__ == '__main__':
    pass