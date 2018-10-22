from atom.api import Atom, ContainerList, observe


class ContoursList(Atom):
    items = ContainerList()
    selectedItems = ContainerList()

    def toggle_item(self, item, toggled):
        if toggled:
            self.selectedItems.append(item)
        else:
            self.selectedItems.remove(item)

    def toggle_all(self, toggled):
        if toggled:
            self.selectedItems = self.items
        else:
            self.selectedItems = []
