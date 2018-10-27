from atom.api import Atom, ContainerList, observe


class ContoursList(Atom):
    items = ContainerList()
    selected_items = ContainerList()

    def toggle_item(self, item, toggled):
        if toggled:
            self.selected_items.append(item)
        else:
            self.selected_items.remove(item)

    def toggle_all(self, toggled):
        if toggled:
            self.selected_items = self.items
        else:
            self.selected_items = []

    def items_at_point(self, x, y):
        return [contour for contour in self.items if contour.point_test(x, y)]
