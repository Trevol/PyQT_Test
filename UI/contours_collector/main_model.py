from atom.api import Atom, ContainerList, Int, Str


class MainModelItem(Atom):
    id = Int()
    name = Str()

    def __init__(self, id, name):
        self.id = id
        self.name = name


class MainModel(Atom):
    items = ContainerList(item=MainModelItem,
                          default=[MainModelItem(1, 'Name 1'), MainModelItem(2, 'Name 2'), MainModelItem(3, 'Name 3')])
    selectedItems = ContainerList(item=MainModelItem)

    def __init__(self):
        self.selectedItems.append(self.items[1])

    def add_item(self):
        nextId = (max([i.id for i in self.items]) + 1) if len(self.items) else 1
        self.items.append(MainModelItem(nextId, f'Name {nextId}'))
