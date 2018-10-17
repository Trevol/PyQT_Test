import functools



count = 0


@functools.lru_cache(128)
def test():
    global count
    count = count + 1
    return count


print(test(), test())


class TestClass:
    count = 0

    def __init__(self, count):
        self.count = count

    @functools.lru_cache(128)
    def test(self):
        self.count += 1
        return self.count


t2 = TestClass(2)
t3 = TestClass(3)

print(t2.test(), t2.test(), t2.test())
print(t3.test(), t3.test(), t3.test())

