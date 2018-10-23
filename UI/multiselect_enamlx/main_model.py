from atom.api import Atom, ContainerList, Typed, Instance

class MainModel(Atom):
    items = ContainerList()
    checked_items = ContainerList()
    selected_item = Typed(object)

    def toggle_item(self, item, toggled):
        if toggled:
            self.checked_items.append(item)
        else:
            self.checked_items.remove(item)


if __name__ == '__main__':
    pass